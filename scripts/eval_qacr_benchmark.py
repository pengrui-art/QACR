#!/usr/bin/env python3
"""End-to-End Evaluation Script for QACR and Baselines on VQA datasets."""

import argparse
import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qacr.data.vqa_dataset import VQADataset
from qacr.eval_utils import compute_vqa_accuracy, postprocess_generation
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_eval_system_prompt(dataset_name: str, textvqa_prompt_mode: str = "v2") -> str:
    base = (
        "Output only the final answer span. "
        "Do not include reasoning, role labels, the question text, or extra explanation. "
        "Do not rewrite the answer into a sentence."
    )
    if dataset_name == "docvqa":
        return (
            "Answer using only the short final answer. "
            "Do not include reasoning, role labels, or extra explanation. "
            "For document questions, copy the exact answer span from the page whenever possible. "
            "Preserve capitalization, decimals, dates, currency, separators, and punctuation exactly."
        )
    if dataset_name == "textvqa":
        if textvqa_prompt_mode == "relaxed":
            return (
                base
                + " For scene-text questions, return the shortest correct answer text. "
                + "Copy visible text exactly when the question asks about the text itself. "
                + "Answer yes or no only for yes/no questions."
            )
        if textvqa_prompt_mode == "shortspan":
            return (
                base
                + " For scene-text questions, return the minimal visible text span that fully answers the question. "
                + "Preserve capitalization, digits, and punctuation when they are part of the answer. "
                + "Answer yes or no only for yes/no questions."
            )
        if textvqa_prompt_mode == "hybrid":
            return (
                base
                + " For scene-text questions, use exact copying for numbers, times, codes, and direct reading questions. "
                + "For brand, place, or sign questions, return the shortest answer span instead of a longer phrase. "
                + "Answer yes or no only for yes/no questions."
            )
        return (
            base
            + " For scene-text questions, copy the visible answer text exactly when possible. "
            + "Preserve capitalization and punctuation when they are part of the visible answer. "
            + "Answer yes or no only for yes/no questions."
        )
    if dataset_name in {"mmmu", "mmbench"}:
        return base + " For multiple-choice questions, answer with only the option letter."
    return base


def build_prompt_question(
    question: str,
    sample: dict,
    dataset_name: str,
    textvqa_prompt_mode: str = "v2",
) -> str:
    options = sample.get("options")
    question_type = str(sample.get("question_type", "open"))
    q = question.lower().strip()
    if question_type == "multiple-choice" and options:
        option_lines = [f"{key}. {value}" for key, value in options.items()]
        return (
            f"{question}\nOptions:\n" + "\n".join(option_lines) + "\nAnswer with only the option letter."
        )
    if dataset_name == "docvqa":
        hint = "Return only the exact answer text from the document."
        if q.startswith("who") or "name of" in q or "to whom" in q:
            hint += " If the answer is a person or organization, copy the full name, including initials and punctuation."
        elif any(k in q for k in ["what time", "which time", "when", "date"]):
            hint += " Copy the exact time or date string, including separators and am/pm."
        elif any(k in q for k in ["how many", "what number", "how much", "what amount", "value", "total amount", "year"]):
            hint += " Copy the exact number, decimals, currency symbols, and units if present."
        elif any(k in q for k in ["website", "web site", "online address", "url", "email", "e-mail"]):
            hint += " Copy the full website or email address exactly."
        elif "address" in q:
            hint += " Copy the full address exactly, including commas and periods."
        return (
            f"{question}\n"
            f"{hint} Keep punctuation exactly if it appears in the document."
        )
    if dataset_name == "textvqa":
        hint = (
            "Read the visible text carefully and return only the answer text. "
            "Do not repeat the question. Do not describe the image."
        )
        if textvqa_prompt_mode == "relaxed":
            hint = (
                "Read the image carefully and return the shortest correct answer. "
                "Use visible text when needed, but avoid extra surrounding words."
            )
        elif textvqa_prompt_mode == "shortspan":
            hint = (
                "Read the visible text carefully and return the minimal answer span. "
                "Do not include extra surrounding words."
            )
        elif textvqa_prompt_mode == "hybrid":
            hint = (
                "Read the visible text carefully and return only the needed answer span. "
                "Keep exact numbers, times, codes, and prices, but avoid extra surrounding words for names, brands, and places."
            )
        if q.startswith(("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "should ", "would ", "will ", "has ", "have ", "had ")):
            hint += " Answer with only yes or no."
        if any(k in q for k in ["how many", "what number", "how much", "price", "what time"]):
            hint += " If the answer is numeric, copy the full number exactly."
        elif q.startswith("who") or "brand" in q or "what does" in q or "name" in q or "store" in q:
            hint += " Copy only the visible text span exactly as written, not a sentence."
        elif "what month" in q:
            hint += " If the image shows an abbreviated month, you may answer with the full month name."
        elif "what state" in q:
            hint += " Return only the state name, not the full sentence around it."
        if textvqa_prompt_mode in {"shortspan", "hybrid"}:
            if "what does" in q or "brand" in q or "where" in q or "name" in q or "store" in q:
                hint += " Prefer the shortest span that directly answers the question."
        if textvqa_prompt_mode == "relaxed":
            hint += " Prefer a short direct answer over copying a whole phrase."
        return (
            f"{question}\n"
            f"{hint} Do not paraphrase."
        )
    return question


def eval_collate_fn(
    batch,
    processor,
    dataset_name="vqav2",
    system_prompt=None,
    textvqa_prompt_mode="v2",
):
    """Collate for evaluation (generation). Only includes User message."""
    if system_prompt is None:
        system_prompt = build_eval_system_prompt(dataset_name, textvqa_prompt_mode=textvqa_prompt_mode)

    images = []
    texts = []
    answers = []
    answer_lists = []
    questions = []
    ocr_tokens_batch = []
    sample_ids = []
    question_types = []
    
    # Optional image transform for low_res baseline
    image_transform = getattr(processor, "_custom_image_transform", None)

    for sample in batch:
        sample_images = sample.get("images") or [sample["image"]]
        processed_images = []
        for image in sample_images:
            if image_transform is not None:
                image = image_transform(image)
            processed_images.append(image)

        question = sample["question"]
        answer = sample["answer"]
        answer_list = sample.get("answers", [answer])
        prompt_question = build_prompt_question(
            question,
            sample,
            dataset_name,
            textvqa_prompt_mode=textvqa_prompt_mode,
        )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": image} for image in processed_images]
                    + [{"type": "text", "text": prompt_question}]
                ),
            },
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)
        images.append(processed_images)
        answers.append(answer)
        answer_lists.append(answer_list)
        questions.append(question)
        ocr_tokens_batch.append(sample.get("ocr_tokens", []))
        sample_ids.append(sample.get("sample_id"))
        question_types.append(sample.get("question_type"))

    messages_for_vision = []
    for image_list in images:
        messages_for_vision.append(
            [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image", "image": img} for img in image_list]
                        + [{"type": "text", "text": "placeholder"}]
                    ),
                }
            ]
        )
    all_image_inputs = []
    for msg_list in messages_for_vision:
        img_in, _ = process_vision_info(msg_list)
        all_image_inputs.extend(img_in)

    inputs = processor(
        text=texts,
        images=all_image_inputs,
        return_tensors="pt",
        padding=True,
    )
    
    inputs["_answers"] = answers
    inputs["_answer_lists"] = answer_lists
    inputs["_questions"] = questions
    inputs["_ocr_tokens"] = ocr_tokens_batch
    inputs["_sample_ids"] = sample_ids
    inputs["_question_types"] = question_types
    return dict(inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to checkpoint directory containing best.pt")
    parser.add_argument("--model", type=str, default="Model/Qwen35-08B", help="Path to Qwen3.5-VL")
    parser.add_argument("--dataset", type=str, default="vqav2", choices=["vqav2", "textvqa", "pope", "docvqa", "mmmu"])
    parser.add_argument(
        "--textvqa-prompt-mode",
        type=str,
        default="v2",
        choices=["v2", "relaxed", "shortspan", "hybrid"],
        help="Prompt variant for TextVQA sweeps. Ignored by other datasets.",
    )
    parser.add_argument("--local-data-dir", type=str, default="/data1/pengrui/CCFA/QACR/data")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit eval samples (e.g. 500 for fast eval)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--min-keep-ratio",
        type=float,
        default=0.0,
        help="Inference-time lower bound on non-skip tokens during hard budget matching.",
    )
    parser.add_argument(
        "--min-deep-ratio",
        type=float,
        default=0.0,
        help="Inference-time lower bound on deep-routed tokens during hard budget matching.",
    )
    parser.add_argument(
        "--executor-output-alpha",
        type=float,
        default=None,
        help="Override executor output blend alpha in [0,1]. None means use checkpoint/default.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per evaluation process")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch_factor (only used when num_workers > 0)",
    )
    parser.add_argument(
        "--no-persistent-workers",
        action="store_true",
        help="Disable DataLoader persistent_workers",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable DataLoader pin_memory",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temp (0 for greedy)")
    parser.add_argument("--out-file", type=str, default=None, help="Optional JSON output path. Defaults to <checkpoint-dir>/eval_results.json")
    parser.add_argument(
        "--protection-mode",
        type=str,
        default="checkpoint",
        choices=["checkpoint", "none", "prior_only", "aux_only", "prior_aux"],
        help="Query-conditioned key-token protection mode. 'checkpoint' reuses saved config when available.",
    )
    parser.add_argument("--protection-topk-scale", type=float, default=None)
    parser.add_argument("--protection-keep-scale", type=float, default=None)
    parser.add_argument("--protection-deep-scale", type=float, default=None)
    parser.add_argument("--protection-logit-bias", type=float, default=None)
    parser.add_argument("--lambda-key-token", type=float, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_dir) / "best.pt"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Loading VLM from %s ...", args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    logger.info("Loading QACR components from %s ...", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Determine type of model from checkpoint
    baseline_type = ckpt.get("baseline", "unknown")
    is_baseline = baseline_type != "unknown"

    from qacr.vision import DepthMultiPathExecutor
    
    if is_baseline:
        logger.info(f"Detected Baseline model: {baseline_type}")
        executor = DepthMultiPathExecutor(
            token_dim=1024,
            hidden_dim=ckpt.get("executor_hidden", 256),
            deep_layers=ckpt.get("deep_layers", 3),
            output_alpha=ckpt.get("executor_output_alpha", 1.0),
        ).to(device)
        executor.load_state_dict(ckpt["executor"])
        if args.executor_output_alpha is not None:
            executor.output_alpha = float(args.executor_output_alpha)
            logger.info("Override executor_output_alpha = %.3f", executor.output_alpha)
        executor.eval()
        
        router = None
        if baseline_type == "image_only":
            from qacr.routing.image_only_router import ImageOnlyRouter
            router = ImageOnlyRouter(image_dim=1024, hidden_dim=128).to(device)
            router.load_state_dict(ckpt["router"])
            router.eval()
            
        from scripts.train_baselines_e2e import BaselineRoutingHook
        
        # Read parameters from string label
        keep_ratio = 0.45
        budget = 0.45
        dir_name = Path(args.checkpoint_dir).name
        if baseline_type == "token_pruning" and "kr" in dir_name:
            kr_str = dir_name.split("kr")[-1].split("_")[0]
            keep_ratio = float(kr_str)
            logger.info(f"Parsed keep_ratio = {keep_ratio}")
        if baseline_type == "image_only" and "b" in dir_name:
            b_str = dir_name.split("_b")[-1].split("_")[0]
            budget = float(b_str)
            logger.info(f"Parsed budget = {budget}")
        if baseline_type == "low_res" and "g" in dir_name:
            g_str = dir_name.split("_g")[-1].split("_")[0]
            grid = int(g_str)
            logger.info(f"Parsed grid = {grid}")
            processor._custom_image_transform = lambda img: img.resize((grid * 28, grid * 28))
            hook_lowres_compute = float((grid * grid) / float(14 * 14))
        else:
            hook_lowres_compute = None

        hook = BaselineRoutingHook(
            baseline=baseline_type,
            executor=executor,
            router=router,
            keep_ratio=keep_ratio,
        )
        if baseline_type == "image_only":
            hook.budget = budget
            hook.temperature = 1e-6 # Hard routing discrete during eval
        if baseline_type == "low_res" and hook_lowres_compute is not None:
            hook.stats.expected_compute = hook_lowres_compute
            
    else:
        logger.info("Detected QACR Query-Adaptive model")
        budget = ckpt.get("budget", 0.45)
        logger.info(f"Loaded QACR with budget: {budget}")
        from qacr.qacr_model import build_qacr_components, QACRRoutingHook
        router, executor = build_qacr_components(
            hidden_dim=1024,
            router_hidden=ckpt.get("router_hidden", 128),
            executor_hidden=ckpt.get("executor_hidden", 256),
            deep_layers=ckpt.get("deep_layers", 3),
            executor_output_alpha=ckpt.get("executor_output_alpha", 1.0),
            device=str(device),
        )
        router.load_state_dict(ckpt["router"])
        executor.load_state_dict(ckpt["executor"])
        if args.executor_output_alpha is not None:
            executor.output_alpha = float(args.executor_output_alpha)
            logger.info("Override executor_output_alpha = %.3f", executor.output_alpha)
        router.to(device).eval()
        executor.to(device).eval()
        
        hook = QACRRoutingHook(
            router=router,
            executor=executor,
            lambda_compute=0.0,
            lambda_entropy=0.0,
        )
        hook.budget = budget
        hook.temperature = 1e-6
        hook.hard_inference = True
        hook.hard_budget_match = True
        hook.min_keep_ratio = args.min_keep_ratio
        hook.min_deep_ratio = args.min_deep_ratio
        hook.dataset_name = args.dataset
        if args.protection_mode == "checkpoint":
            hook.protection_mode = str(ckpt.get("protection_mode", "none"))
        else:
            hook.protection_mode = args.protection_mode
        hook.protection_topk_scale = float(
            ckpt.get("protection_topk_scale", 1.0)
            if args.protection_topk_scale is None
            else args.protection_topk_scale
        )
        hook.protection_keep_scale = float(
            ckpt.get("protection_keep_scale", 1.0)
            if args.protection_keep_scale is None
            else args.protection_keep_scale
        )
        hook.protection_deep_scale = float(
            ckpt.get("protection_deep_scale", 1.0)
            if args.protection_deep_scale is None
            else args.protection_deep_scale
        )
        hook.protection_logit_bias = float(
            ckpt.get("protection_logit_bias", 1.0)
            if args.protection_logit_bias is None
            else args.protection_logit_bias
        )
        hook.lambda_key_token = float(
            ckpt.get("lambda_key_token", 0.0)
            if args.lambda_key_token is None
            else args.lambda_key_token
        )

    visual_encoder_merger = model.model.visual.merger
    handle = visual_encoder_merger.register_forward_hook(hook)

    logger.info("Loading evaluation dataset ...")
    ds = VQADataset(
        dataset_name=args.dataset,
        split="eval",
        max_samples=args.max_samples,
        streaming=False,
        local_dir=args.local_data_dir,
    )
    collate = lambda b: eval_collate_fn(
        b,
        processor,
        dataset_name=args.dataset,
        textvqa_prompt_mode=args.textvqa_prompt_mode,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": not args.no_pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = not args.no_persistent_workers
    loader = DataLoader(ds, **loader_kwargs)

    results = []
    total_acc = 0.0
    total_raw_acc = 0.0
    total_compute = 0.0
    total_skip = 0.0
    total_deep = 0.0
    total_samples = 0

    logger.info("Starting Evaluation ...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Eval")):
            input_ids = batch.pop("input_ids").to(device)
            # Remove keys that generate() doesn't expect or use
            gt_answers = batch.pop("_answers")
            gt_answer_lists = batch.pop("_answer_lists")
            questions = batch.pop("_questions")
            ocr_tokens_batch = batch.pop("_ocr_tokens")
            sample_ids = batch.pop("_sample_ids")
            question_types = batch.pop("_question_types")
            
            # For QACR Query embedding hook
            if not is_baseline:
                query_embeds = model.get_input_embeddings()(input_ids).float()
                hook.query_embeds = query_embeds
                hook.grid_thw = batch["image_grid_thw"].to(device)
                hook.questions = questions
                hook.question_types = question_types
                hook.ocr_tokens_batch = ocr_tokens_batch
            
            # Send other required keys to device
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Generate
            outputs = model.generate(
                input_ids=input_ids,
                **batch,
                max_new_tokens=32,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None,
            )
            
            # Hook stats were populated during the prefill forward pass
            batch_compute = getattr(hook.stats, 'expected_compute', 0.0)
            batch_skip = getattr(hook.stats, 'mean_skip', 0.0)
            batch_deep = getattr(hook.stats, 'mean_deep', 0.0)

            if is_baseline and baseline_type == "low_res" and hook_lowres_compute is not None:
                batch_compute = hook_lowres_compute

            # Process generation outputs
            generated_ids = outputs[:, input_ids.shape[1]:]  # skip prompt tokens
            pred_strs = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            for pred_str, gt_str, gt_list, question, ocr_tokens, sample_id, question_type in zip(
                pred_strs,
                gt_answers,
                gt_answer_lists,
                questions,
                ocr_tokens_batch,
                sample_ids,
                question_types,
            ):
                clean_pred = postprocess_generation(
                    pred_str,
                    question=question,
                    dataset_name=args.dataset,
                    ocr_tokens=ocr_tokens,
                )
                acc = compute_vqa_accuracy(
                    clean_pred,
                    gt_list,
                    dataset_name=args.dataset,
                )
                raw_acc = compute_vqa_accuracy(
                    pred_str,
                    gt_list,
                    dataset_name=args.dataset,
                )
                total_acc += acc
                total_raw_acc += raw_acc
                total_samples += 1
                
                results.append({
                    "sample_id": sample_id,
                    "question": question,
                    "question_type": question_type,
                    "gt": gt_str,
                    "gt_answers": gt_list,
                    "pred_raw": pred_str,
                    "pred": clean_pred,
                    "ocr_tokens": ocr_tokens,
                    "acc": acc,
                    "raw_acc": raw_acc,
                })
                
            total_compute += batch_compute * len(pred_strs)
            total_skip += batch_skip * len(pred_strs)
            total_deep += batch_deep * len(pred_strs)

    handle.remove()
    
    metrics = {
        "accuracy": total_acc / total_samples,
        "raw_accuracy": total_raw_acc / total_samples,
        "mean_compute": total_compute / total_samples,
        "mean_skip_ratio": total_skip / total_samples,
        "mean_deep_ratio": total_deep / total_samples,
        "total_evaluated": total_samples,
    }
    logger.info("\n========== Benchmark Results ==========\n" + \
                "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items()) + \
                "\n=======================================")

    out_file = Path(args.out_file) if args.out_file else Path(args.checkpoint_dir) / "eval_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "metrics": metrics,
            "args": vars(args),
            "results": results,
        }, f, indent=2)
    logger.info("Saved detailed results to %s", out_file)


if __name__ == "__main__":
    main()
