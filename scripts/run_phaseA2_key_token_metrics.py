#!/usr/bin/env python3
"""Phase A.2: compute prediction-side key-token metrics from eval outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qacr.analysis import (
    annotate_key_tokens,
    compute_prediction_key_token_metrics,
)
from qacr.data.vqa_dataset import DATASET_CONFIGS, _extract_answer_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["textvqa", "docvqa"])
    parser.add_argument("--local-data-dir", type=str, default="data")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method spec in the form name=path/to/result.json . Repeatable.",
    )
    parser.add_argument("--min-results", type=int, default=100, help="Skip files with fewer evaluated samples.")
    parser.add_argument("--report-examples", type=int, default=8)
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Defaults to outputs/phaseA_key_token_metrics/<dataset>_key_token_metrics.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default=None,
        help="Defaults to outputs/phaseA_key_token_metrics/<dataset>_key_token_metrics.md",
    )
    return parser.parse_args()


def parse_method_specs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --method spec: {spec}")
        name, path = spec.split("=", 1)
        parsed.append((name.strip(), Path(path.strip())))
    return parsed


def load_results(path: Path, min_results: int) -> dict | None:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    total = int(metrics.get("total_evaluated", len(data.get("results", []))))
    if total < min_results:
        return None
    return data


def build_dataset_index(dataset_name: str, local_data_dir: str, max_samples: int) -> list[dict]:
    from datasets import load_dataset
    import os

    cfg = DATASET_CONFIGS[dataset_name]
    hf_split = cfg["eval_split"]
    config_name = cfg.get("config_name")

    dataset_folder_name = cfg["hf_name"].split("/")[-1]
    if os.path.isdir(os.path.join(local_data_dir, dataset_folder_name)):
        source = os.path.join(local_data_dir, dataset_folder_name)
    else:
        source = local_data_dir

    load_kwargs = {"trust_remote_code": False}
    if config_name is not None:
        ds = load_dataset(source, config_name, split=hf_split, **load_kwargs)
    else:
        ds = load_dataset(source, split=hf_split, **load_kwargs)

    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    records: list[dict] = []
    answers_key = cfg.get("answers_key", cfg["answer_key"])
    question_type_key = cfg.get("question_type_key")
    for idx, row in enumerate(ds):
        answers = _extract_answer_list(row.get(answers_key, row[cfg["answer_key"]]))
        qtype = row.get(question_type_key) if question_type_key is not None else None
        if isinstance(qtype, list):
            qtype = [str(item) for item in qtype if item is not None]
        sample = {
            "dataset_name": dataset_name,
            "sample_id": str(
                row.get("questionId")
                or row.get("question_id")
                or row.get("sample_id")
                or row.get("id")
                or f"{dataset_name}:{idx}"
            ),
            "question": str(row[cfg["question_key"]]),
            "answers": answers,
            "question_type": qtype,
        }
        if "ocr_tokens" in row:
            sample["ocr_tokens"] = [str(token) for token in row.get("ocr_tokens", []) if token is not None]
        elif "ocr" in row and row.get("ocr"):
            sample["ocr_tokens"] = [tok for tok in str(row["ocr"]).split() if tok]
        else:
            sample["ocr_tokens"] = []
        records.append(sample)
    return records


def _check_alignment(result_entry: dict, sample: dict) -> bool:
    q_res = str(result_entry.get("question", "")).strip()
    q_sample = str(sample.get("question", "")).strip()
    return q_res == q_sample


def evaluate_method(name: str, data: dict, dataset_samples: list[dict]) -> dict:
    results = data["results"]
    if len(results) > len(dataset_samples):
        raise ValueError(f"{name}: results length {len(results)} exceeds dataset length {len(dataset_samples)}")

    aggregates = defaultdict(float)
    type_buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    examples: list[dict] = []
    alignment_mismatch = 0

    total_results = len(results)

    for idx, result_entry in enumerate(results):
        sample = dataset_samples[idx]
        if not _check_alignment(result_entry, sample):
            alignment_mismatch += 1

        annotation = annotate_key_tokens(
            question=sample["question"],
            gt_answers=sample.get("answers", result_entry.get("gt_answers", [])),
            dataset_name=sample.get("dataset_name", data.get("args", {}).get("dataset", "")) or "",
            ocr_tokens=sample.get("ocr_tokens", result_entry.get("ocr_tokens", [])),
        )
        pred_metrics = compute_prediction_key_token_metrics(result_entry.get("pred", ""), annotation)
        raw_metrics = compute_prediction_key_token_metrics(result_entry.get("pred_raw", ""), annotation)

        qtype = annotation["question_type"]
        aggregates["total_count"] += 1
        aggregates["token_level_sum"] += float(annotation["protocol_level"] == "token")
        aggregates["with_key_tokens_sum"] += float(bool(annotation["key_token_indices"]))

        if pred_metrics["target_unit_count"] > 0:
            aggregates["count"] += 1
            aggregates["pred_key_token_recall_sum"] += pred_metrics["key_token_recall"]
            aggregates["pred_key_token_miss_rate_sum"] += pred_metrics["key_token_miss_rate"]
            aggregates["pred_all_target_units_hit_sum"] += float(pred_metrics["all_target_units_hit"])
            aggregates["pred_any_target_unit_hit_sum"] += float(pred_metrics["any_target_unit_hit"])
            aggregates["raw_key_token_recall_sum"] += raw_metrics["key_token_recall"]
            aggregates["raw_key_token_miss_rate_sum"] += raw_metrics["key_token_miss_rate"]

            bucket = type_buckets[qtype]
            bucket["count"] += 1
            bucket["pred_key_token_recall_sum"] += pred_metrics["key_token_recall"]
            bucket["pred_all_target_units_hit_sum"] += float(pred_metrics["all_target_units_hit"])

        if pred_metrics["target_unit_count"] > 0 and pred_metrics["key_token_recall"] < 1.0:
            examples.append({
                "sample_index": idx,
                "sample_id": sample.get("sample_id"),
                "question": sample["question"],
                "answers": sample.get("answers", [])[:3],
                "pred": result_entry.get("pred", ""),
                "pred_raw": result_entry.get("pred_raw", ""),
                "question_type": qtype,
                "protocol_level": annotation["protocol_level"],
                "match_strategy": annotation["match_strategy"],
                "key_tokens": annotation["key_tokens"][:12],
                "answer_units": annotation["answer_units"],
                "pred_key_token_recall": pred_metrics["key_token_recall"],
                "pred_target_unit_count": pred_metrics["target_unit_count"],
                "pred_matched_target_units": pred_metrics["matched_target_units"],
            })

    count = int(aggregates["count"])
    summary = {
        "method": name,
        "num_results": total_results,
        "num_measurable_results": count,
        "alignment_mismatch_count": int(alignment_mismatch),
        "pred_key_token_recall": float(aggregates["pred_key_token_recall_sum"] / count) if count else 0.0,
        "pred_key_token_miss_rate": float(aggregates["pred_key_token_miss_rate_sum"] / count) if count else 0.0,
        "pred_all_target_units_hit_rate": float(aggregates["pred_all_target_units_hit_sum"] / count) if count else 0.0,
        "pred_any_target_unit_hit_rate": float(aggregates["pred_any_target_unit_hit_sum"] / count) if count else 0.0,
        "raw_key_token_recall": float(aggregates["raw_key_token_recall_sum"] / count) if count else 0.0,
        "raw_key_token_miss_rate": float(aggregates["raw_key_token_miss_rate_sum"] / count) if count else 0.0,
        "token_level_ratio": float(aggregates["token_level_sum"] / total_results) if total_results else 0.0,
        "samples_with_key_tokens_ratio": float(aggregates["with_key_tokens_sum"] / total_results) if total_results else 0.0,
        "question_type_breakdown": {},
    }
    for qtype, bucket in sorted(type_buckets.items()):
        bcount = int(bucket["count"])
        summary["question_type_breakdown"][qtype] = {
            "count": bcount,
            "pred_key_token_recall": float(bucket["pred_key_token_recall_sum"] / bcount) if bcount else 0.0,
            "pred_all_target_units_hit_rate": float(bucket["pred_all_target_units_hit_sum"] / bcount) if bcount else 0.0,
        }

    examples.sort(key=lambda x: (x["pred_key_token_recall"], x["pred_matched_target_units"], x["pred_target_unit_count"]))
    return {"summary": summary, "examples": examples}


def render_markdown(dataset: str, report: dict[str, dict], report_examples: int) -> str:
    lines: list[str] = []
    lines.append(f"# Phase A.2 Key-Token Metrics Report ({dataset})")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This is the first usable A.2 version and measures **prediction-side key-token preservation**, not route-level coverage.")
    lines.append("- It reuses the Phase A.1 labeling protocol and aligns eval results with dataset samples by evaluation order.")
    lines.append("- Route-level metrics can be added later once per-sample route dumps are available.")
    lines.append("")
    lines.append("## Method Summary")
    lines.append("")
    lines.append("| Method | N | Measurable N | Pred Key Recall | Pred Miss Rate | All Units Hit | Any Unit Hit | Raw Key Recall | Token-Level Ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, section in report.items():
        s = section["summary"]
        lines.append(
            f"| {name} | {s['num_results']} | {s['num_measurable_results']} | {s['pred_key_token_recall']:.4f} | {s['pred_key_token_miss_rate']:.4f} | "
            f"{s['pred_all_target_units_hit_rate']:.4f} | {s['pred_any_target_unit_hit_rate']:.4f} | "
            f"{s['raw_key_token_recall']:.4f} | {s['token_level_ratio']:.4f} |"
        )
    lines.append("")

    for name, section in report.items():
        s = section["summary"]
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- alignment_mismatch_count: `{s['alignment_mismatch_count']}`")
        lines.append(f"- num_measurable_results: `{s['num_measurable_results']}`")
        lines.append(f"- samples_with_key_tokens_ratio: `{s['samples_with_key_tokens_ratio']:.4f}`")
        lines.append("")
        lines.append("### Question-Type Breakdown")
        lines.append("")
        lines.append("| Question Type | Count | Pred Key Recall | All Units Hit |")
        lines.append("|---|---:|---:|---:|")
        for qtype, bucket in s["question_type_breakdown"].items():
            lines.append(
                f"| {qtype} | {bucket['count']} | {bucket['pred_key_token_recall']:.4f} | {bucket['pred_all_target_units_hit_rate']:.4f} |"
            )
        lines.append("")
        lines.append("### Representative Miss Cases")
        lines.append("")
        for example in section["examples"][:report_examples]:
            lines.append(f"- `sample_id={example['sample_id']}`")
            lines.append(f"  - question: {example['question']}")
            lines.append(f"  - answers: {example['answers']}")
            lines.append(f"  - pred: {example['pred']}")
            lines.append(f"  - pred_raw: {example['pred_raw']}")
            lines.append(f"  - question_type: {example['question_type']}")
            lines.append(f"  - protocol_level: {example['protocol_level']}")
            lines.append(f"  - match_strategy: {example['match_strategy']}")
            lines.append(f"  - key_tokens: {example['key_tokens']}")
            lines.append(f"  - answer_units: {example['answer_units']}")
            lines.append(
                f"  - pred_key_token_recall: {example['pred_key_token_recall']:.4f} "
                f"({example['pred_matched_target_units']}/{example['pred_target_unit_count']})"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    method_specs = parse_method_specs(args.method)
    if not method_specs:
        raise ValueError("At least one --method name=path spec is required")

    out_json = Path(args.out_json or f"outputs/phaseA_key_token_metrics/{args.dataset}_key_token_metrics.json")
    out_md = Path(args.out_md or f"outputs/phaseA_key_token_metrics/{args.dataset}_key_token_metrics.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    loaded_methods: list[tuple[str, dict]] = []
    max_results = 0
    for name, path in method_specs:
        data = load_results(path, args.min_results)
        if data is None:
            print(f"Skipping {name}: fewer than {args.min_results} evaluated samples in {path}")
            continue
        loaded_methods.append((name, data))
        max_results = max(max_results, len(data["results"]))
    if not loaded_methods:
        raise ValueError("No usable result files after min-results filtering")

    dataset_samples = build_dataset_index(args.dataset, args.local_data_dir, max_results)

    report: dict[str, dict] = {}
    for name, data in loaded_methods:
        report[name] = evaluate_method(name, data, dataset_samples)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with out_md.open("w", encoding="utf-8") as f:
        f.write(render_markdown(args.dataset, report, args.report_examples))

    print(f"Saved JSON report to {out_json}")
    print(f"Saved Markdown report to {out_md}")


if __name__ == "__main__":
    main()
