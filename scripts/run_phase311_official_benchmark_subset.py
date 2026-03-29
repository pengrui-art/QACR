#!/usr/bin/env python3
"""Task 3.11: run official benchmark subsets under a unified QACR protocol."""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import re
import string
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from qwen_vl_utils import process_vision_info
from torchvision import transforms
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qacr.routing import (
    DepthOnlyRouter,
    compute_regularization_loss,
    linear_temperature,
    soft_routing_probs,
)
from qacr.vision import DepthMultiPathExecutor


PUNCT_TABLE = str.maketrans("", "", string.punctuation)
ARTICLES = {"a", "an", "the"}
ROUTE_COSTS = torch.tensor([0.0, 0.35, 1.0], dtype=torch.float32)
METHODS = ("LowRes", "TokenPruning", "QACR-DepthOnly")
DATASETS = ("VQAv2", "GQA", "POPE", "TextVQA", "DocVQA", "MMBench", "MMMU")


@dataclass
class BenchmarkSample:
    dataset: str
    sample_id: str
    question: str
    images: list[Image.Image]
    answers: list[str]
    task_type: str
    options: dict[str, str] | None = None
    hint: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--budgets", default="0.35,0.45,0.60")
    parser.add_argument("--subset-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--router-hidden", type=int, default=96)
    parser.add_argument("--executor-hidden", type=int, default=128)
    parser.add_argument("--deep-layers", type=int, default=3)
    parser.add_argument("--coarse-grid", type=int, default=14)
    parser.add_argument("--router-steps", type=int, default=10)
    parser.add_argument("--router-lr", type=float, default=2e-4)
    parser.add_argument("--lambda-compute", type=float, default=0.8)
    parser.add_argument("--lambda-entropy", type=float, default=0.02)
    parser.add_argument("--temp-start", type=float, default=1.5)
    parser.add_argument("--temp-end", type=float, default=0.4)
    parser.add_argument(
        "--out-json", default="outputs/phase311_official_benchmark_subset_results.json"
    )
    parser.add_argument(
        "--out-md", default="outputs/phase311_official_benchmark_subset_results.md"
    )
    parser.add_argument(
        "--pareto-compute-png",
        default="outputs/phase311_accuracy_compute_pareto.png",
    )
    parser.add_argument(
        "--pareto-latency-png",
        default="outputs/phase311_accuracy_latency_pareto.png",
    )
    parser.add_argument(
        "--temp-image-dir", default="outputs/phase311_transformed_images"
    )
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_budgets(text: str) -> list[float]:
    budgets = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not budgets:
        raise ValueError("At least one budget is required")
    return budgets


def normalize_text(text: str) -> str:
    text = text.lower().translate(PUNCT_TABLE)
    tokens = [tok for tok in text.split() if tok and tok not in ARTICLES]
    return " ".join(tokens)


def image_dict_to_pil(image_dict: dict[str, Any]) -> Image.Image:
    image_bytes = image_dict.get("bytes")
    if image_bytes is None:
        raise ValueError("Expected image bytes in dataset sample")
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def array_to_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(v) for v in value.tolist() if str(v).strip()]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def load_parquet(repo_id: str, filename: str) -> pd.DataFrame:
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return pd.read_parquet(local_path)


def load_vqav2_samples(limit: int) -> list[BenchmarkSample]:
    df = load_parquet("merve/vqav2-small", "data/validation-00000-of-00007.parquet")
    samples: list[BenchmarkSample] = []
    for idx, row in df.head(limit).iterrows():
        samples.append(
            BenchmarkSample(
                dataset="VQAv2",
                sample_id=f"vqav2_{idx}",
                question=str(row["question"]),
                images=[image_dict_to_pil(row["image"])],
                answers=[str(row["multiple_choice_answer"])],
                task_type="open",
            )
        )
    return samples


def load_gqa_samples(limit: int) -> list[BenchmarkSample]:
    instructions = load_parquet(
        "lmms-lab/GQA",
        "testdev_balanced_instructions/testdev-00000-of-00001.parquet",
    )
    images = load_parquet(
        "lmms-lab/GQA",
        "testdev_balanced_images/testdev-00000-of-00001.parquet",
    )
    image_map = {str(row["id"]): row["image"] for _, row in images.iterrows()}
    samples: list[BenchmarkSample] = []
    for idx, row in instructions.head(limit).iterrows():
        image_key = str(row["imageId"])
        if image_key not in image_map:
            continue
        samples.append(
            BenchmarkSample(
                dataset="GQA",
                sample_id=f"gqa_{idx}",
                question=str(row["question"]),
                images=[image_dict_to_pil(image_map[image_key])],
                answers=[str(row["answer"])],
                task_type="open",
            )
        )
    return samples


def load_pope_samples(limit: int) -> list[BenchmarkSample]:
    df = load_parquet("lmms-lab/POPE", "data/test-00000-of-00003.parquet")
    samples: list[BenchmarkSample] = []
    for idx, row in df.head(limit).iterrows():
        samples.append(
            BenchmarkSample(
                dataset="POPE",
                sample_id=f"pope_{idx}",
                question=str(row["question"]),
                images=[image_dict_to_pil(row["image"])],
                answers=[str(row["answer"])],
                task_type="yesno",
            )
        )
    return samples


def load_textvqa_samples(limit: int) -> list[BenchmarkSample]:
    df = load_parquet(
        "Multimodal-Fatima/TextVQA_validation",
        "data/validation-00000-of-00003-8690d90f45ce561f.parquet",
    )
    samples: list[BenchmarkSample] = []
    for idx, row in df.head(limit).iterrows():
        samples.append(
            BenchmarkSample(
                dataset="TextVQA",
                sample_id=f"textvqa_{idx}",
                question=str(row["question"]),
                images=[image_dict_to_pil(row["image"])],
                answers=array_to_answers(row["answers"]),
                task_type="open",
            )
        )
    return samples


def load_docvqa_samples(limit: int) -> list[BenchmarkSample]:
    df = load_parquet(
        "lmms-lab/DocVQA", "DocVQA/validation-00000-of-00006.parquet"
    )
    samples: list[BenchmarkSample] = []
    for idx, row in df.head(limit).iterrows():
        samples.append(
            BenchmarkSample(
                dataset="DocVQA",
                sample_id=f"docvqa_{idx}",
                question=str(row["question"]),
                images=[image_dict_to_pil(row["image"])],
                answers=array_to_answers(row["answers"]),
                task_type="open",
            )
        )
    return samples


def load_mmbench_samples(limit: int) -> list[BenchmarkSample]:
    df = load_parquet(
        "HuggingFaceM4/MMBench_dev",
        "data/train-00000-of-00001-28992cf4da792fdc.parquet",
    )
    samples: list[BenchmarkSample] = []
    label_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    for idx, row in df.head(limit).iterrows():
        options = {}
        for key in ("A", "B", "C", "D"):
            value = row.get(key)
            if is_missing_value(value):
                continue
            options[key] = str(value)
        label = row["label"]
        answer = label_map.get(int(label), str(label))
        samples.append(
            BenchmarkSample(
                dataset="MMBench",
                sample_id=f"mmbench_{idx}",
                question=str(row["question"]),
                hint=str(row["hint"]) if not is_missing_value(row["hint"]) else None,
                images=[image_dict_to_pil(row["image"])],
                answers=[answer],
                task_type="multiple_choice",
                options=options,
            )
        )
    return samples


def load_mmmu_samples(limit: int) -> list[BenchmarkSample]:
    df = load_parquet("lmms-lab/MMMU", "data/validation-00000-of-00001.parquet")
    samples: list[BenchmarkSample] = []
    for idx, row in df.head(limit).iterrows():
        raw_options = ast.literal_eval(str(row["options"]))
        option_letters = "ABCDEFG"
        options = {
            option_letters[i]: str(option)
            for i, option in enumerate(raw_options)
            if i < len(option_letters)
        }
        images = []
        for image_key in [f"image_{i}" for i in range(1, 8)]:
            image_value = row.get(image_key)
            if is_missing_value(image_value):
                continue
            images.append(image_dict_to_pil(image_value))
        samples.append(
            BenchmarkSample(
                dataset="MMMU",
                sample_id=f"mmmu_{idx}",
                question=str(row["question"]),
                images=images,
                answers=[str(row["answer"])],
                task_type="multiple_choice",
                options=options,
            )
        )
    return samples


def load_samples(dataset_names: list[str], limit: int) -> tuple[list[BenchmarkSample], dict[str, str]]:
    loaders = {
        "VQAv2": load_vqav2_samples,
        "GQA": load_gqa_samples,
        "POPE": load_pope_samples,
        "TextVQA": load_textvqa_samples,
        "DocVQA": load_docvqa_samples,
        "MMBench": load_mmbench_samples,
        "MMMU": load_mmmu_samples,
    }
    all_samples: list[BenchmarkSample] = []
    failures: dict[str, str] = {}
    for dataset_name in dataset_names:
        try:
            all_samples.extend(loaders[dataset_name](limit))
        except Exception as exc:  # pragma: no cover - runtime data loading guard
            failures[dataset_name] = f"{type(exc).__name__}: {exc}"
    return all_samples, failures


def sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pick_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def pil_to_tensor(image: Image.Image, size: int = 448) -> torch.Tensor:
    tfm = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return tfm(image).unsqueeze(0)


def image_to_tokens(image: Image.Image, grid: int) -> torch.Tensor:
    pixel = pil_to_tensor(image)
    pooled = F.adaptive_avg_pool2d(pixel, output_size=(grid, grid))
    return pooled.flatten(2).transpose(1, 2).contiguous()


def tokens_to_image(tokens: torch.Tensor, grid: int, size: int = 448) -> Image.Image:
    if tokens.ndim == 3:
        tokens = tokens[0]
    grid_tensor = tokens.transpose(0, 1).reshape(1, 3, grid, grid)
    upsampled = F.interpolate(
        grid_tensor, size=(size, size), mode="bilinear", align_corners=False
    )
    array = (
        upsampled.squeeze(0)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
    return Image.fromarray((array * 255.0).astype(np.uint8))


def load_query_tokens(
    processor: AutoProcessor, model: AutoModelForImageTextToText, query: str
) -> torch.Tensor:
    tokenized = processor.tokenizer([query], return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"].to(next(model.parameters()).device)
    with torch.no_grad():
        query_tokens = model.get_input_embeddings()(input_ids)
    return query_tokens.detach().float()


def build_pruning_route_indices(image_tokens: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    num_tokens = image_tokens.size(1)
    keep_k = max(1, int(round(num_tokens * keep_ratio)))
    scores = torch.norm(image_tokens, p=2, dim=-1)
    topk_idx = torch.topk(scores, k=keep_k, dim=-1, largest=True).indices
    route_idx = torch.zeros(
        image_tokens.size(0), num_tokens, dtype=torch.long, device=image_tokens.device
    )
    batch_idx = torch.arange(image_tokens.size(0), device=image_tokens.device).unsqueeze(1)
    route_idx[batch_idx, topk_idx] = 2
    return route_idx


def route_indices_to_compute_ratio(route_idx: torch.Tensor, route_costs: torch.Tensor) -> float:
    route_costs = route_costs.to(device=route_idx.device, dtype=torch.float32)
    one_hot = F.one_hot(route_idx, num_classes=3).float()
    value = (one_hot * route_costs.view(1, 1, -1)).sum(dim=-1).mean()
    return float(value.detach().cpu())


def budget_matched_route_indices(
    route_probs: torch.Tensor, budget: float, route_costs: torch.Tensor
) -> torch.Tensor:
    if route_probs.ndim != 3 or route_probs.size(-1) != 3:
        raise ValueError("route_probs must be [B, T, 3]")
    route_costs = route_costs.to(device=route_probs.device, dtype=route_probs.dtype)
    batch_size, num_tokens, _ = route_probs.shape
    output = torch.zeros(batch_size, num_tokens, dtype=torch.long, device=route_probs.device)

    shallow_cost = float(route_costs[1].item())
    deep_cost = float(route_costs[2].item())

    for batch_idx in range(batch_size):
        probs = route_probs[batch_idx]
        target_cost = budget * num_tokens
        used_cost = 0.0
        candidates = []
        for token_idx in range(num_tokens):
            skip_prob = float(probs[token_idx, 0].item())
            shallow_gain = float(probs[token_idx, 1].item()) - skip_prob
            deep_gain = float(probs[token_idx, 2].item()) - skip_prob
            candidates.append(
                (
                    shallow_gain / max(shallow_cost, 1e-6),
                    shallow_gain,
                    shallow_cost,
                    token_idx,
                    1,
                )
            )
            candidates.append(
                (
                    deep_gain / max(deep_cost, 1e-6),
                    deep_gain,
                    deep_cost,
                    token_idx,
                    2,
                )
            )

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        assigned_tokens: set[int] = set()
        for _, _, cost, token_idx, route_id in candidates:
            if token_idx in assigned_tokens:
                continue
            if used_cost + cost > target_cost + 1e-6:
                continue
            output[batch_idx, token_idx] = route_id
            assigned_tokens.add(token_idx)
            used_cost += cost

    return output


def train_executor_for_fixed_routes(
    image_tokens: torch.Tensor,
    route_idx: torch.Tensor,
    hidden_dim: int,
    deep_layers: int,
    steps: int,
    lr: float,
    device: torch.device,
) -> tuple[DepthMultiPathExecutor, torch.Tensor]:
    executor = DepthMultiPathExecutor(
        token_dim=image_tokens.size(-1),
        hidden_dim=hidden_dim,
        deep_layers=deep_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(executor.parameters(), lr=lr)
    fixed_one_hot = F.one_hot(route_idx, num_classes=3).float()

    for _ in range(steps):
        pred, _ = executor(
            image_tokens=image_tokens,
            route_probs=fixed_one_hot,
            route_indices=route_idx,
            mode="hard",
        )
        loss = F.mse_loss(pred, image_tokens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred, _ = executor(
            image_tokens=image_tokens,
            route_probs=fixed_one_hot,
            route_indices=route_idx,
            mode="hard_conditional",
        )
    return executor, pred


def run_qacr_transform(
    image_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    budget: float,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    router = DepthOnlyRouter(
        query_dim=query_tokens.size(-1),
        image_dim=image_tokens.size(-1),
        hidden_dim=args.router_hidden,
    ).to(device)
    executor = DepthMultiPathExecutor(
        token_dim=image_tokens.size(-1),
        hidden_dim=args.executor_hidden,
        deep_layers=args.deep_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(router.parameters()) + list(executor.parameters()),
        lr=args.router_lr,
    )
    route_costs = ROUTE_COSTS.to(device=device, dtype=image_tokens.dtype)
    final_soft_probs = None

    for step in range(args.router_steps):
        temp = linear_temperature(step, args.router_steps, args.temp_start, args.temp_end)
        router_out = router(query_tokens=query_tokens, image_tokens=image_tokens)
        soft_probs = soft_routing_probs(router_out.logits, temperature=temp, use_gumbel=True)
        pred, _ = executor(image_tokens=image_tokens, route_probs=soft_probs, mode="soft")
        task_loss = F.mse_loss(pred, image_tokens)
        compute_loss, _ = compute_regularization_loss(
            route_probs=soft_probs, route_costs=route_costs, budget_ratio=budget
        )
        entropy = -(soft_probs * soft_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        loss = task_loss + args.lambda_compute * compute_loss - args.lambda_entropy * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_soft_probs = soft_probs.detach()

    if final_soft_probs is None:
        raise RuntimeError("QACR transform did not produce routing probabilities")

    hard_idx = budget_matched_route_indices(final_soft_probs, budget=budget, route_costs=ROUTE_COSTS)
    hard_one_hot = F.one_hot(hard_idx, num_classes=3).float()
    with torch.no_grad():
        pred, _ = executor(
            image_tokens=image_tokens,
            route_probs=hard_one_hot,
            route_indices=hard_idx,
            mode="hard_conditional",
        )
    compute_ratio = route_indices_to_compute_ratio(hard_idx, ROUTE_COSTS)
    return pred, compute_ratio


def lowres_transform_image(image: Image.Image, budget: float, base_grid: int) -> tuple[Image.Image, float]:
    target_grid = max(2, int(round(base_grid * math.sqrt(budget))))
    actual_ratio = float((target_grid * target_grid) / (base_grid * base_grid))
    resized = image.resize((target_grid, target_grid), resample=Image.BICUBIC)
    restored = resized.resize((448, 448), resample=Image.BICUBIC)
    return restored, actual_ratio


def transform_image_for_method(
    method: str,
    image: Image.Image,
    question: str,
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Image.Image, float]:
    if method == "LowRes":
        return lowres_transform_image(image=image, budget=args.current_budget, base_grid=args.coarse_grid)

    image_tokens = image_to_tokens(image, grid=args.coarse_grid).to(device)

    if method == "TokenPruning":
        route_idx = build_pruning_route_indices(image_tokens, keep_ratio=args.current_budget).to(device)
        _, pred = train_executor_for_fixed_routes(
            image_tokens=image_tokens,
            route_idx=route_idx,
            hidden_dim=args.executor_hidden,
            deep_layers=args.deep_layers,
            steps=max(4, args.router_steps // 2),
            lr=args.router_lr,
            device=device,
        )
        compute_ratio = route_indices_to_compute_ratio(route_idx, ROUTE_COSTS)
        return tokens_to_image(pred, grid=args.coarse_grid), compute_ratio

    if method == "QACR-DepthOnly":
        query_tokens = load_query_tokens(processor, model, question).to(device)
        pred, compute_ratio = run_qacr_transform(
            image_tokens=image_tokens,
            query_tokens=query_tokens,
            budget=args.current_budget,
            args=args,
            device=device,
        )
        return tokens_to_image(pred, grid=args.coarse_grid), compute_ratio

    raise ValueError(f"Unknown method: {method}")


def save_transformed_images(
    images: list[Image.Image],
    method: str,
    sample: BenchmarkSample,
    budget: float,
    temp_dir: Path,
) -> list[Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    budget_tag = str(budget).replace(".", "p")
    for idx, image in enumerate(images):
        out_path = temp_dir / f"{sample.dataset}_{sample.sample_id}_{method}_{budget_tag}_{idx}.png"
        image.save(out_path)
        saved_paths.append(out_path)
    return saved_paths


def build_prompt(sample: BenchmarkSample) -> str:
    if sample.task_type == "multiple_choice":
        option_lines = [f"{key}. {value}" for key, value in (sample.options or {}).items()]
        parts = []
        if sample.hint:
            parts.append(sample.hint)
        parts.append(sample.question)
        parts.append("选项如下：")
        parts.extend(option_lines)
        parts.append("请只回答选项字母。")
        return "\n".join(parts)
    if sample.task_type == "yesno":
        return f"{sample.question}\n请只回答是或否。"
    return sample.question


def run_model_inference(
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    image_paths: list[Path],
    prompt: str,
    max_new_tokens: int,
) -> str:
    content: list[dict[str, Any]] = []
    for image_path in image_paths:
        content.append({"type": "image", "image": str(image_path.resolve())})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    model_device = next(model.parameters()).device
    inputs = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def judge_yes_no(text: str) -> bool | None:
    lowered = text.strip().lower()
    yes_markers = ["是", "有", "yes", "true"]
    no_markers = ["否", "没有", "无", "not", "no", "false"]
    yes_hit = any(marker in lowered for marker in yes_markers)
    no_hit = any(marker in lowered for marker in no_markers)
    if yes_hit and not no_hit:
        return True
    if no_hit and not yes_hit:
        return False
    return None


def extract_choice_letter(text: str, options: dict[str, str] | None) -> str | None:
    match = re.search(r"\b([A-G])\b", text.upper())
    if match:
        return match.group(1)
    normalized_text = normalize_text(text)
    if options:
        for key, value in options.items():
            normalized_value = normalize_text(str(value))
            if normalized_value and normalized_value in normalized_text:
                return key
    return None


def score_open_response(response: str, answers: list[str]) -> float:
    pred = normalize_text(response)
    refs = [normalize_text(ans) for ans in answers if normalize_text(ans)]
    if not pred or not refs:
        return 0.0
    for ref in refs:
        if pred == ref or pred in ref or ref in pred:
            return 1.0
    return 0.0


def score_sample(sample: BenchmarkSample, response: str) -> float:
    if sample.task_type == "yesno":
        pred = judge_yes_no(response)
        target = judge_yes_no(sample.answers[0])
        return float(pred is not None and target is not None and pred == target)
    if sample.task_type == "multiple_choice":
        pred = extract_choice_letter(response, sample.options)
        return float(pred is not None and pred == sample.answers[0])
    return score_open_response(response, sample.answers)


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    per_dataset_grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(row["method"], row["budget"])].append(row)
        per_dataset_grouped[(row["dataset"], row["method"], row["budget"])].append(row)

    summary_rows = []
    for (method, budget), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "method": method,
                "budget": budget,
                "num_samples": len(rows),
                "avg_accuracy": float(np.mean([row["accuracy"] for row in rows])),
                "avg_compute_ratio": float(np.mean([row["compute_ratio"] for row in rows])),
                "avg_latency_ms": float(np.mean([row["latency_ms"] for row in rows])),
                "avg_peak_memory_mb": float(np.mean([row["peak_memory_mb"] for row in rows])),
            }
        )

    per_dataset_rows = []
    for (dataset, method, budget), rows in sorted(per_dataset_grouped.items()):
        per_dataset_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "budget": budget,
                "num_samples": len(rows),
                "avg_accuracy": float(np.mean([row["accuracy"] for row in rows])),
                "avg_compute_ratio": float(np.mean([row["compute_ratio"] for row in rows])),
                "avg_latency_ms": float(np.mean([row["latency_ms"] for row in rows])),
                "avg_peak_memory_mb": float(np.mean([row["peak_memory_mb"] for row in rows])),
            }
        )

    return {
        "macro_summary": summary_rows,
        "per_dataset_summary": per_dataset_rows,
        "matched_compute_table": sorted(
            summary_rows, key=lambda row: (row["budget"], -row["avg_accuracy"], row["avg_latency_ms"])
        ),
        "matched_latency_table": sorted(
            summary_rows, key=lambda row: (row["avg_latency_ms"], -row["avg_accuracy"])
        ),
    }


def export_markdown(
    aggregate: dict[str, Any],
    failures: dict[str, str],
    out_md: Path,
) -> None:
    lines = [
        "# Phase 3.11 Official Benchmark Subset Results",
        "",
        "## Macro Summary",
        "",
        "| Method | Budget | Samples | Avg Accuracy | Avg Compute | Avg Latency (ms) | Avg Peak Mem (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate["macro_summary"]:
        lines.append(
            "| {method} | {budget:.2f} | {num_samples} | {avg_accuracy:.4f} | "
            "{avg_compute_ratio:.4f} | {avg_latency_ms:.4f} | {avg_peak_memory_mb:.2f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Per-Dataset Summary",
            "",
            "| Dataset | Method | Budget | Samples | Avg Accuracy | Avg Compute | Avg Latency (ms) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate["per_dataset_summary"]:
        lines.append(
            "| {dataset} | {method} | {budget:.2f} | {num_samples} | {avg_accuracy:.4f} | "
            "{avg_compute_ratio:.4f} | {avg_latency_ms:.4f} |".format(**row)
        )

    if failures:
        lines.extend(["", "## Load Failures", ""])
        for dataset_name, error_text in failures.items():
            lines.append(f"- {dataset_name}: {error_text}")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def plot_pareto(rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for row in rows:
        label = f"{row['method']}@{row['budget']:.2f}"
        plt.scatter(row[x_key], row[y_key], s=70)
        plt.text(row[x_key], row[y_key], label, fontsize=8)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    selected_datasets = parse_csv_list(args.datasets)
    budgets = parse_budgets(args.budgets)
    samples, failures = load_samples(selected_datasets, limit=args.subset_size)
    if not samples:
        raise RuntimeError("No benchmark samples loaded")

    dtype = pick_dtype()
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_dir = Path(args.temp_image_dir)
    raw_results: list[dict[str, Any]] = []

    print("===== Phase 3.11 Official Benchmark Subsets =====")
    print(f"datasets_loaded: {sorted({sample.dataset for sample in samples})}")
    print("dataset | method | budget | accuracy | compute_ratio | latency_ms | peak_mem_mb")

    for sample in samples:
        prompt = build_prompt(sample)
        for budget in budgets:
            args.current_budget = budget
            for method in METHODS:
                sync_if_cuda()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()

                transformed_images = []
                compute_ratios = []
                for image in sample.images:
                    transformed_image, compute_ratio = transform_image_for_method(
                        method=method,
                        image=image,
                        question=sample.question,
                        processor=processor,
                        model=model,
                        args=args,
                        device=device,
                    )
                    transformed_images.append(transformed_image)
                    compute_ratios.append(compute_ratio)

                image_paths = save_transformed_images(
                    images=transformed_images,
                    method=method,
                    sample=sample,
                    budget=budget,
                    temp_dir=temp_dir,
                )
                response = run_model_inference(
                    processor=processor,
                    model=model,
                    image_paths=image_paths,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                sync_if_cuda()
                latency_ms = (time.perf_counter() - start) * 1000.0
                peak_memory_mb = (
                    float(torch.cuda.max_memory_allocated() / (1024**2))
                    if torch.cuda.is_available()
                    else 0.0
                )
                accuracy = score_sample(sample, response)
                result = {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "method": method,
                    "budget": budget,
                    "accuracy": float(accuracy),
                    "compute_ratio": float(np.mean(compute_ratios)),
                    "latency_ms": float(latency_ms),
                    "peak_memory_mb": peak_memory_mb,
                    "response": response,
                    "answers": sample.answers,
                }
                raw_results.append(result)
                print(
                    f"{sample.dataset} | {method} | {budget:.2f} | {accuracy:.4f} | "
                    f"{result['compute_ratio']:.4f} | {latency_ms:.4f} | {peak_memory_mb:.2f}"
                )
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    aggregate = aggregate_results(raw_results)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "task": "3.11_official_benchmark_subsets",
                "model": str(model_path),
                "datasets_requested": selected_datasets,
                "datasets_loaded": sorted({sample.dataset for sample in samples}),
                "subset_size_per_dataset": args.subset_size,
                "budgets": budgets,
                "methods": list(METHODS),
                "failures": failures,
                "aggregate": aggregate,
                "raw_results": raw_results,
                "note": (
                    "Accuracy is measured on official benchmark subsets under the current "
                    "RGB-token proxy executor; latency and peak memory use a unified end-to-end "
                    "protocol including image transformation and Qwen generation."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    export_markdown(aggregate=aggregate, failures=failures, out_md=Path(args.out_md))
    plot_pareto(
        rows=aggregate["macro_summary"],
        x_key="avg_compute_ratio",
        y_key="avg_accuracy",
        title="Phase 3.11 Accuracy-Compute Pareto",
        out_path=Path(args.pareto_compute_png),
    )
    plot_pareto(
        rows=aggregate["macro_summary"],
        x_key="avg_latency_ms",
        y_key="avg_accuracy",
        title="Phase 3.11 Accuracy-Latency Pareto",
        out_path=Path(args.pareto_latency_png),
    )

    print(f"summary_json: {out_json}")
    print(f"summary_md: {args.out_md}")
    print(f"pareto_compute_png: {args.pareto_compute_png}")
    print(f"pareto_latency_png: {args.pareto_latency_png}")
    print("status: official benchmark subset evaluation succeeded")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
