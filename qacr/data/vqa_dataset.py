"""Unified VQA dataset for QACR training and evaluation.

Supports VQAv2, TextVQA, POPE, DocVQA, and MMMU from HuggingFace or local mirrors.
"""

from __future__ import annotations

import logging
import ast
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── Dataset field mapping ──────────────────────────────────────────────────────
# Each HF dataset uses slightly different column names for images/questions/answers.
DATASET_CONFIGS = {
    "vqav2": {
        "hf_name": "lmms-lab/VQAv2",
        # lmms-lab/VQAv2 has: validation / testdev / test  (no train split).
        # IMPORTANT: Only 'validation' split has ground-truth answers.
        # 'testdev' and 'test' have multiple_choice_answer=None → always use 'validation'.
        "train_split": "validation",
        "eval_split": "validation",  # DO NOT change to testdev/test — they have no answers!
        "image_key": "image",
        "question_key": "question",
        "answer_key": "multiple_choice_answer",
        "answers_key": "answers",
    },
    "textvqa": {
        "hf_name": "lmms-lab/textvqa",
        "train_split": "train",
        "eval_split": "validation",
        "image_key": "image",
        "question_key": "question",
        "answer_key": "answers",          # list → take first
        "answers_key": "answers",
    },
    "pope": {
        "hf_name": "lmms-lab/POPE",
        "train_split": None,               # POPE has no train split
        "eval_split": "test",
        "image_key": "image",
        "question_key": "question",
        "answer_key": "answer",
        "answers_key": "answer",
    },
    "docvqa": {
        "hf_name": "lmms-lab/DocVQA",
        "config_name": "DocVQA",
        "train_split": "train",
        "eval_split": "validation",
        "image_key": "image",
        "question_key": "question",
        "answer_key": "answers",
        "answers_key": "answers",
    },
    "mmmu": {
        "hf_name": "lmms-lab/MMMU",
        "train_split": "dev",
        "eval_split": "validation",
        "image_keys": [f"image_{i}" for i in range(1, 8)],
        "question_key": "question",
        "answer_key": "answer",
        "answers_key": "answer",
        "options_key": "options",
        "question_type_key": "question_type",
    },
}


def _extract_answer(raw_answer: Any) -> str:
    """Normalise heterogeneous answer formats to a single string."""
    if raw_answer is None:
        return ""  # Guard: testdev/test splits have None answers — return empty string
    if isinstance(raw_answer, list):
        # Filter out None entries first
        valid = [a for a in raw_answer if a is not None]
        return str(valid[0]) if valid else ""
    return str(raw_answer)


def _extract_answer_list(raw_answers: Any) -> list[str]:
    """Normalise heterogeneous answer annotations to a flat string list."""
    if raw_answers is None:
        return []
    if isinstance(raw_answers, list):
        answers: list[str] = []
        for item in raw_answers:
            if item is None:
                continue
            if isinstance(item, dict):
                value = item.get("answer")
                if value is not None:
                    answers.append(str(value))
            else:
                answers.append(str(item))
        return answers
    if isinstance(raw_answers, dict):
        value = raw_answers.get("answer")
        return [str(value)] if value is not None else []
    return [str(raw_answers)]


def _extract_images(row: dict[str, Any], image_key: str | None, image_keys: list[str] | None) -> list[Any]:
    if image_keys:
        images = []
        for key in image_keys:
            value = row.get(key)
            if value is None:
                continue
            images.append(value)
        return images
    if image_key is None:
        return []
    return [row[image_key]]


def _extract_options(raw_options: Any) -> dict[str, str] | None:
    if raw_options is None:
        return None
    if isinstance(raw_options, dict):
        return {str(k): str(v) for k, v in raw_options.items()}
    if isinstance(raw_options, list):
        letters = "ABCDEFG"
        return {letters[i]: str(v) for i, v in enumerate(raw_options) if i < len(letters)}
    if isinstance(raw_options, str):
        try:
            parsed = ast.literal_eval(raw_options)
        except Exception:
            return None
        return _extract_options(parsed)
    return None


class VQADataset(Dataset):
    """Wraps a HuggingFace VQA‑style dataset for QACR training/eval.

    Parameters
    ----------
    dataset_name : str
        One of ``"vqav2"``, ``"textvqa"``, ``"pope"``, ``"docvqa"``, ``"mmmu"``.
    split : str
        ``"train"`` or ``"eval"`` (mapped to the HF split automatically).
    max_samples : int | None
        Cap the number of samples (useful for quick experiments).
    streaming : bool
        If True, use HF streaming (no full download required).
    local_dir : str | None
        If set, load parquet files from this local directory instead of HF Hub.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: int | None = None,
        streaming: bool = False,
        local_dir: str | None = None,
    ) -> None:
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Choose from {list(DATASET_CONFIGS)}"
            )

        cfg = DATASET_CONFIGS[dataset_name]
        hf_split = cfg["train_split"] if split == "train" else cfg["eval_split"]
        if hf_split is None:
            raise ValueError(f"Dataset '{dataset_name}' has no '{split}' split")

        self.dataset_name = dataset_name
        self.image_key = cfg.get("image_key")
        self.image_keys = cfg.get("image_keys")
        self.question_key = cfg["question_key"]
        self.answer_key = cfg["answer_key"]
        self.answers_key = cfg.get("answers_key", self.answer_key)
        self.options_key = cfg.get("options_key")
        self.question_type_key = cfg.get("question_type_key")
        config_name = cfg.get("config_name")

        # ── Load dataset ───────────────────────────────────────────────────
        from datasets import load_dataset

        load_kwargs: dict[str, Any] = {"trust_remote_code": False}
        if local_dir is not None:
            import os
            dataset_folder_name = cfg["hf_name"].split("/")[-1]
            if os.path.isdir(os.path.join(local_dir, dataset_folder_name)):
                source = os.path.join(local_dir, dataset_folder_name)
            else:
                source = local_dir
        else:
            source = cfg["hf_name"]

        # IMPORTANT: Always explicitly pass the split name to load_dataset.
        # This ensures we never accidentally load 'testdev' (which has no answers).
        # hf_split is derived from DATASET_CONFIGS and is always 'validation' for vqav2.
        if streaming:
            if config_name is not None:
                ds = load_dataset(source, config_name, split=hf_split, streaming=True, **load_kwargs)
            else:
                ds = load_dataset(source, split=hf_split, streaming=True, **load_kwargs)
            # Materialise up to max_samples from the stream
            n = max_samples or 5000
            logger.info("Streaming %s/%s, collecting up to %d samples …", source, hf_split, n)
            rows: list[dict] = []
            for i, row in enumerate(ds):
                if i >= n:
                    break
                rows.append(row)
            self.samples = rows
            logger.info("Collected %d samples via streaming.", len(self.samples))
        else:
            if config_name is not None:
                ds = load_dataset(source, config_name, split=hf_split, **load_kwargs)
            else:
                ds = load_dataset(source, split=hf_split, **load_kwargs)
            logger.info("Raw dataset loaded: %s/%s, total=%d", source, hf_split, len(ds))

            if dataset_name == "vqav2":
                # VQAv2 validation has 214354 samples.
                # Convention: first 20000 → train partition, rest → eval partition.
                # This avoids train/eval overlap while staying within the same split
                # (the only split that has ground-truth answers).
                ds = ds.shuffle(seed=42)
                if split == "train":
                    train_size = min(20000, len(ds))
                    end_idx = train_size + max_samples if max_samples is not None else train_size
                    ds = ds.select(range(0, min(end_idx, train_size)))
                else:  # eval
                    train_size = min(20000, len(ds))
                    end_idx = train_size + max_samples if max_samples is not None else len(ds)
                    ds = ds.select(range(train_size, min(end_idx, len(ds))))
            elif max_samples is not None and max_samples < len(ds):
                ds = ds.shuffle(seed=42).select(range(max_samples))

            self.samples = ds
            logger.info("Loaded %d samples from %s/%s (split=%s).", len(self.samples), source, hf_split, split)

    # ── Dataset interface ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]
        images = _extract_images(row, self.image_key, self.image_keys)
        question = row[self.question_key]       # str
        answer = _extract_answer(row[self.answer_key])
        answers = _extract_answer_list(row.get(self.answers_key, row[self.answer_key]))
        if not answers and answer:
            answers = [answer]
        sample = {
            "image": images[0] if images else None,
            "images": images,
            "question": question,
            "answer": answer,
            "answers": answers,
            "sample_id": str(
                row.get("questionId")
                or row.get("question_id")
                or row.get("sample_id")
                or row.get("id")
                or f"{self.dataset_name}:{idx}"
            ),
        }
        if "ocr_tokens" in row:
            sample["ocr_tokens"] = [str(token) for token in row.get("ocr_tokens", []) if token is not None]
        if self.options_key is not None:
            sample["options"] = _extract_options(row.get(self.options_key))
        if self.question_type_key is not None:
            value = row.get(self.question_type_key, "open")
            if isinstance(value, list):
                sample["question_type"] = [str(item) for item in value if item is not None]
            else:
                sample["question_type"] = str(value)
        return sample


# ── Collation helper ───────────────────────────────────────────────────────────

def vqa_collate_fn(
    batch: list[dict[str, Any]],
    processor: Any,
    system_prompt: str = "Answer the question concisely.",
) -> dict[str, Any]:
    """Collate a list of VQA samples into model‑ready tensors.

    Builds Qwen‑VL chat messages, tokenises with the processor, and creates
    ``labels`` that mask everything *before* the assistant answer with ``-100``.
    """
    images = []
    texts = []
    answers = []
    questions = []
    question_types = []
    ocr_tokens_batch = []

    for sample in batch:
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]

        # ── build chat messages (with answer for teacher‑forcing) ──────────
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        images.append(image)
        answers.append(answer)
        questions.append(question)
        question_types.append(sample.get("question_type"))
        ocr_tokens_batch.append(sample.get("ocr_tokens", []))

    # ── tokenise & build pixel tensors ─────────────────────────────────────
    from qwen_vl_utils import process_vision_info

    messages_for_vision = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "placeholder"},
                ],
            }
        ]
        for img in images
    ]
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

    # ── create labels: mask everything except assistant answer ─────────────
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()

    # Find the assistant header token sequence and mask everything before it
    # For Qwen, the assistant section starts after "<|im_start|>assistant\n"
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    for b in range(input_ids.size(0)):
        ids = input_ids[b].tolist()
        # Find the LAST occurrence of <|im_start|> (= assistant header)
        last_im_start = -1
        for pos in range(len(ids) - 1, -1, -1):
            if ids[pos] == im_start_id:
                last_im_start = pos
                break
        if last_im_start >= 0:
            # Mask: system + user + assistant header (up to \\n after "assistant")
            # Find the newline after "assistant"
            mask_end = last_im_start + 3  # <|im_start|> + "assistant" + \n
            mask_end = min(mask_end, input_ids.size(1))
            labels[b, :mask_end] = -100
        # Also mask padding
        pad_id = processor.tokenizer.pad_token_id or 0
        labels[b, input_ids[b] == pad_id] = -100

    inputs["labels"] = labels
    inputs["_answers"] = answers  # keep for eval reference
    inputs["_questions"] = questions
    inputs["_question_types"] = question_types
    inputs["_ocr_tokens"] = ocr_tokens_batch
    return dict(inputs)
