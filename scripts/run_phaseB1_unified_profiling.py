#!/usr/bin/env python3
"""Phase B.1: unified end-to-end latency / throughput / memory profiling."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qacr.data.vqa_dataset import VQADataset
from qacr.qacr_model import QACRRoutingHook, build_qacr_components
from qacr.vision import DepthMultiPathExecutor
from qacr.routing.image_only_router import ImageOnlyRouter
from scripts.eval_qacr_benchmark import eval_collate_fn
from scripts.train_baselines_e2e import BaselineRoutingHook
from transformers import AutoModelForImageTextToText, AutoProcessor


METHOD_REGISTRY = {
    "original": {
        "label": "Original",
        "kind": "original",
        "checkpoint": None,
    },
    "qacr_b045": {
        "label": "QACR b0.45",
        "kind": "qacr",
        "checkpoint": "checkpoints/qacr_vqav2_b0.45/best.pt",
    },
    "token_pruning": {
        "label": "TokenPruning@0.45",
        "kind": "baseline",
        "checkpoint": "checkpoints/token_pruning_kr0.45_vqav2/best.pt",
    },
    "image_only": {
        "label": "ImageOnly@0.45",
        "kind": "baseline",
        "checkpoint": "checkpoints/image_only_b0.45_vqav2/best.pt",
    },
    "low_res": {
        "label": "LowRes-9x9",
        "kind": "baseline",
        "checkpoint": "checkpoints/low_res_g9_vqav2/best.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Model/Qwen35-08B")
    parser.add_argument("--dataset", type=str, default="textvqa", choices=["textvqa", "docvqa", "mmmu", "vqav2", "pope"])
    parser.add_argument("--local-data-dir", type=str, default="data")
    parser.add_argument("--methods", type=str, default="original,qacr_b045,token_pruning,image_only,low_res")
    parser.add_argument("--batch-sizes", type=str, default="1,2")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--out-json",
        type=str,
        default="outputs/phaseB_unified_profiling/phaseB1_unified_profiling.json",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="outputs/phaseB_unified_profiling/phaseB1_unified_profiling.csv",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="outputs/phaseB_unified_profiling/phaseB1_unified_profiling.md",
    )
    return parser.parse_args()


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_cuda(*objs) -> None:
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_csv_arg(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_model_and_processor(model_path: str, gpu_id: int):
    device_map = f"cuda:{gpu_id}" if torch.cuda.is_available() else None
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token_id is not None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


def build_loader(args: argparse.Namespace, processor, batch_size: int):
    ds = VQADataset(
        dataset_name=args.dataset,
        split="eval",
        max_samples=args.max_samples,
        streaming=False,
        local_dir=args.local_data_dir,
    )
    collate = lambda b: eval_collate_fn(b, processor, dataset_name=args.dataset)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": not args.no_pin_memory,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = False
    return ds, DataLoader(ds, **kwargs)


def setup_method(args: argparse.Namespace, method_key: str, model):
    spec = METHOD_REGISTRY[method_key]
    kind = spec["kind"]
    hook = None
    handle_owner = None
    configured_compute = 1.0
    ckpt = None
    processor_image_transform = None

    if kind == "original":
        return {
            "hook": None,
            "configured_compute": 1.0,
            "processor_image_transform": None,
            "label": spec["label"],
        }

    ckpt_path = Path(spec["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    if kind == "qacr":
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
        router.eval()
        executor.eval()
        hook = QACRRoutingHook(router=router, executor=executor, lambda_compute=0.0, lambda_entropy=0.0)
        hook.budget = ckpt.get("budget", 0.45)
        hook.temperature = 1e-6
        hook.hard_inference = True
        hook.hard_budget_match = True
        configured_compute = float(ckpt.get("budget", 0.45))
        handle_owner = model.model.visual.merger
    else:
        baseline_type = ckpt.get("baseline", "unknown")
        executor = DepthMultiPathExecutor(
            token_dim=1024,
            hidden_dim=ckpt.get("executor_hidden", 256),
            deep_layers=ckpt.get("deep_layers", 3),
            output_alpha=ckpt.get("executor_output_alpha", 1.0),
        ).to(device)
        executor.load_state_dict(ckpt["executor"])
        executor.eval()

        router = None
        if baseline_type == "image_only":
            router = ImageOnlyRouter(image_dim=1024, hidden_dim=128).to(device)
            router.load_state_dict(ckpt["router"])
            router.eval()

        keep_ratio = 0.45
        if baseline_type == "token_pruning":
            keep_ratio = 0.45
        hook = BaselineRoutingHook(
            baseline=baseline_type,
            executor=executor,
            router=router,
            keep_ratio=keep_ratio,
        )
        if baseline_type == "image_only":
            hook.budget = 0.45
            hook.temperature = 1e-6
            configured_compute = 0.45
        elif baseline_type == "token_pruning":
            configured_compute = 0.45
        elif baseline_type == "low_res":
            grid = 9
            processor_image_transform = lambda img: img.resize((grid * 28, grid * 28))
            configured_compute = float((grid * grid) / float(14 * 14))
        handle_owner = model.model.visual.merger

    return {
        "hook": hook,
        "configured_compute": configured_compute,
        "processor_image_transform": processor_image_transform,
        "label": spec["label"],
        "handle_owner": handle_owner,
    }


def profile_one_method_batch(args: argparse.Namespace, method_key: str, batch_size: int) -> dict:
    model, processor = load_model_and_processor(args.model, args.gpu_id)
    setup = setup_method(args, method_key, model)
    processor._custom_image_transform = setup["processor_image_transform"]
    _, loader = build_loader(args, processor, batch_size=batch_size)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    hook = setup["hook"]
    handle_owner = setup.get("handle_owner")

    warmup_left = args.warmup_batches
    measured_samples = 0
    measured_batches = 0
    total_compute = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        start_memory_mb = float(torch.cuda.memory_allocated(device) / (1024**2))
    else:
        start_memory_mb = 0.0

    measured_start = None
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch.pop("input_ids").to(device)
            batch.pop("_answers")
            batch.pop("_answer_lists")
            batch.pop("_questions")
            batch.pop("_ocr_tokens")
            batch.pop("_sample_ids")
            batch.pop("_question_types")

            if hook is not None and method_key == "qacr_b045":
                query_embeds = model.get_input_embeddings()(input_ids).float()
                hook.query_embeds = query_embeds
                hook.grid_thw = batch["image_grid_thw"].to(device)

            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            handle = handle_owner.register_forward_hook(hook) if hook is not None else None
            try:
                sync_cuda()
                if warmup_left > 0:
                    _ = model.generate(
                        input_ids=input_ids,
                        **batch,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None,
                    )
                    sync_cuda()
                    warmup_left -= 1
                    if warmup_left == 0 and torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(device)
                        start_memory_mb = float(torch.cuda.memory_allocated(device) / (1024**2))
                    continue

                if measured_start is None:
                    measured_start = time.perf_counter()

                outputs = model.generate(
                    input_ids=input_ids,
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None,
                )
                sync_cuda()
            finally:
                if handle is not None:
                    handle.remove()

            _ = outputs[:, input_ids.shape[1]:]
            batch_n = input_ids.size(0)
            measured_samples += batch_n
            measured_batches += 1
            if method_key == "low_res":
                batch_compute = float(setup["configured_compute"])
            elif hook is not None:
                batch_compute = float(getattr(hook.stats, "expected_compute", setup["configured_compute"]))
            else:
                batch_compute = float(setup["configured_compute"])
            total_compute += batch_compute * batch_n

    sync_cuda()
    measured_end = time.perf_counter()
    elapsed = max((measured_end - measured_start) if measured_start is not None else 0.0, 1e-8)
    if torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    else:
        peak_memory_mb = 0.0

    result = {
        "method_key": method_key,
        "method": setup["label"],
        "dataset": args.dataset,
        "batch_size": batch_size,
        "max_samples": args.max_samples,
        "warmup_batches": args.warmup_batches,
        "num_workers": args.num_workers,
        "measured_samples": measured_samples,
        "measured_batches": measured_batches,
        "wall_clock_latency_ms": float(elapsed * 1000.0 / max(measured_batches, 1)),
        "wall_clock_latency_per_sample_ms": float(elapsed * 1000.0 / max(measured_samples, 1)),
        "throughput_samples_per_s": float(measured_samples / elapsed),
        "mean_compute_profiled": float(total_compute / max(measured_samples, 1)),
        "start_gpu_memory_mb": start_memory_mb,
        "peak_gpu_memory_mb": peak_memory_mb,
        "peak_gpu_memory_delta_mb": float(max(peak_memory_mb - start_memory_mb, 0.0)),
    }

    cleanup_cuda(loader, model, processor, hook)
    return result


def write_outputs(rows: list[dict], out_json: Path, out_csv: Path, out_md: Path) -> None:
    payload = {
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "method_key",
        "method",
        "dataset",
        "batch_size",
        "max_samples",
        "measured_samples",
        "measured_batches",
        "wall_clock_latency_ms",
        "wall_clock_latency_per_sample_ms",
        "throughput_samples_per_s",
        "mean_compute_profiled",
        "start_gpu_memory_mb",
        "peak_gpu_memory_mb",
        "peak_gpu_memory_delta_mb",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)

    lines: list[str] = []
    lines.append("# Phase B.1 Unified Profiling")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Method | Batch | Mean Compute | Batch Latency (ms) | Sample Latency (ms) | Throughput (samples/s) | Peak GPU Memory (MB) | Delta Peak (MB) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['batch_size']} | {row['mean_compute_profiled']:.4f} | "
            f"{row['wall_clock_latency_ms']:.2f} | {row['wall_clock_latency_per_sample_ms']:.2f} | "
            f"{row['throughput_samples_per_s']:.2f} | {row['peak_gpu_memory_mb']:.2f} | {row['peak_gpu_memory_delta_mb']:.2f} |"
        )
    lines.append("")
    lines.append("## Batch-Size Sensitivity")
    lines.append("")
    for method, method_rows in grouped.items():
        method_rows = sorted(method_rows, key=lambda x: x["batch_size"])
        lines.append(f"### {method}")
        lines.append("")
        for row in method_rows:
            lines.append(
                f"- `bs={row['batch_size']}`: latency `{row['wall_clock_latency_ms']:.2f} ms/batch`, "
                f"throughput `{row['throughput_samples_per_s']:.2f} samples/s`, "
                f"peak memory `{row['peak_gpu_memory_mb']:.2f} MB`."
            )
        lines.append("")
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    method_keys = parse_csv_arg(args.methods)
    batch_sizes = parse_int_csv(args.batch_sizes)

    unknown = [key for key in method_keys if key not in METHOD_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown method keys: {unknown}")

    rows: list[dict] = []
    for method_key in method_keys:
        for batch_size in batch_sizes:
            print(f"[profile] method={method_key} batch_size={batch_size}")
            row = profile_one_method_batch(args, method_key, batch_size)
            rows.append(row)
            print(
                f"[done] {row['method']} bs={row['batch_size']} latency={row['wall_clock_latency_ms']:.2f}ms "
                f"throughput={row['throughput_samples_per_s']:.2f}/s peak_mem={row['peak_gpu_memory_mb']:.2f}MB"
            )

    write_outputs(rows, Path(args.out_json), Path(args.out_csv), Path(args.out_md))
    print(f"saved_json: {args.out_json}")
    print(f"saved_csv: {args.out_csv}")
    print(f"saved_md: {args.out_md}")


if __name__ == "__main__":
    main()
