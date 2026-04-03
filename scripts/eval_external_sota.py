#!/usr/bin/env python3
"""External Training-Free SOTA Evaluation Script for Qwen3.5-VL on VQA datasets."""

import argparse
import json
import logging
import os
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
from scripts.eval_qacr_benchmark import eval_collate_fn
from qacr.external_sota_patch import apply_fastv, apply_lvpruning

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Model/Qwen35-08B", help="Path to Qwen3.5-VL")
    parser.add_argument("--dataset", type=str, default="vqav2", choices=["vqav2", "textvqa", "pope", "docvqa", "mmmu"])
    parser.add_argument("--local-data-dir", type=str, default="/data1/pengrui/CCFA/QACR/data")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit eval samples (e.g. 500 for fast eval)")
    parser.add_argument("--batch-size", type=int, default=8)
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
    parser.add_argument("--method", type=str, default="fastv", choices=["fastv", "lvpruning", "original"], help="SOTA method to apply")
    parser.add_argument("--keep-ratio", type=float, default=0.45, help="Keep ratio for FastV")
    parser.add_argument("--k-layer", type=int, default=3, help="Layer K for FastV")
    parser.add_argument("--out-dir", type=str, default="checkpoints/sota_eval", help="Directory to save eval_results.json")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info("Loading VLM from %s ...", args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    if args.method == "fastv":
        logger.info(f"Applying FastV patch at layer {args.k_layer} with keep_ratio {args.keep_ratio}")
        apply_fastv(model, prune_layer_idx=args.k_layer, keep_ratio=args.keep_ratio)
        configured_compute = float(args.keep_ratio)
    elif args.method == "lvpruning":
        logger.info(f"Applying LVPruning patch with default multi-layer settings")
        apply_lvpruning(model)
        configured_compute = 0.45
    else:
        logger.info("No patch applied, running original model.")
        configured_compute = 1.0

    logger.info("Loading evaluation dataset ...")
    ds = VQADataset(
        dataset_name=args.dataset,
        split="eval",
        max_samples=args.max_samples,
        streaming=False,
        local_dir=args.local_data_dir,
    )
    collate = lambda b: eval_collate_fn(b, processor, dataset_name=args.dataset)
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
    total_samples = 0

    logger.info("Starting Evaluation ...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"Eval {args.method}")):
            input_ids = batch.pop("input_ids").to(device)
            gt_answers = batch.pop("_answers")
            gt_answer_lists = batch.pop("_answer_lists")
            questions = batch.pop("_questions")
            
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model.generate(
                input_ids=input_ids,
                **batch,
                max_new_tokens=32,
                do_sample=(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None,
            )
            
            # The original visual token count is the total number of visual tokens
            # For simplistic compute proxy in SOTA patch, compute = pruned / original
            batch_compute = configured_compute
            if args.method != "original" and getattr(model, "_sota_tracked_original_length", 0) > 0:
                # Approximation of compute ratio: 
                # (1) Vision encoder compute is unaffected (1.0)
                # (2) LLM compute is theoretically reduced by the ratio of dropped tokens
                # To be fair in Pareto plots, we just record the keep_ratio as mean_compute for now
                # Or use empirical ratios:
                pruned = model._sota_tracked_pruned_length
                orig = model._sota_tracked_original_length
                # If we consider LLM compute, we can average them. For now, track empirical patch ratio
                tracked_ratio = float(pruned) / float(orig)
                if tracked_ratio > 0:
                    batch_compute = min(batch_compute, tracked_ratio)

            generated_ids = outputs[:, input_ids.shape[1]:]  # skip prompt tokens
            pred_strs = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            for pred_str, gt_str, gt_list, question in zip(pred_strs, gt_answers, gt_answer_lists, questions):
                clean_pred = postprocess_generation(
                    pred_str,
                    question=question,
                    dataset_name=args.dataset,
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
                    "question": question,
                    "gt": gt_str,
                    "gt_answers": gt_list,
                    "pred_raw": pred_str,
                    "pred": clean_pred,
                    "acc": acc,
                    "raw_acc": raw_acc,
                })
                
            total_compute += batch_compute * len(pred_strs)

    metrics = {
        "accuracy": total_acc / total_samples,
        "raw_accuracy": total_raw_acc / total_samples,
        "mean_compute": total_compute / total_samples,
        "total_evaluated": total_samples,
    }
    logger.info("\n========== Benchmark Results ==========\n" + \
                "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items()) + \
                "\n=======================================")

    out_file = Path(args.out_dir) / f"{args.method}_{args.dataset}_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "metrics": metrics,
            "args": vars(args),
            "results": results,
        }, f, indent=2)
    logger.info("Saved detailed results to %s", out_file)


if __name__ == "__main__":
    main()
