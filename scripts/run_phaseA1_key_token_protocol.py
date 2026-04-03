#!/usr/bin/env python3
"""Phase A.1: define and sanity-check the key-token labeling protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qacr.analysis import annotate_key_tokens, summarize_key_token_annotations
from qacr.data.vqa_dataset import VQADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="textvqa,docvqa",
        help="Comma-separated datasets from {textvqa,docvqa}.",
    )
    parser.add_argument("--local-data-dir", type=str, default="data")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--report-samples-per-dataset",
        type=int,
        default=8,
        help="Number of representative examples to store in the markdown report per dataset.",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="outputs/phaseA_key_token_protocol/phaseA1_key_token_protocol.json",
    )
    parser.add_argument(
        "--out-md",
        type=str,
        default="outputs/phaseA_key_token_protocol/phaseA1_key_token_protocol.md",
    )
    return parser.parse_args()


def _sample_to_record(sample: dict, dataset_name: str, sample_idx: int) -> dict:
    annotation = annotate_key_tokens(
        question=sample["question"],
        gt_answers=sample.get("answers", []),
        dataset_name=dataset_name,
        ocr_tokens=sample.get("ocr_tokens", []),
    )
    return {
        "sample_index": sample_idx,
        "sample_id": sample.get("sample_id", f"{dataset_name}:{sample_idx}"),
        "question": sample["question"],
        "answers": sample.get("answers", []),
        "ocr_tokens": sample.get("ocr_tokens", []),
        **annotation,
    }


def _pick_examples(records: list[dict], limit: int) -> list[dict]:
    def sort_key(record: dict) -> tuple[int, int, int]:
        token_bonus = 1 if record.get("protocol_level") == "token" else 0
        strategy_bonus = 1 if record.get("match_strategy") == "ocr_span_match" else 0
        key_count = len(record.get("key_token_indices", []))
        return (-token_bonus, -strategy_bonus, -key_count)

    return sorted(records, key=sort_key)[:limit]


def _render_markdown(report: dict[str, dict], datasets: list[str]) -> str:
    lines: list[str] = []
    lines.append("# Phase A.1 Key-Token Protocol Report")
    lines.append("")
    lines.append("## Protocol Summary")
    lines.append("")
    lines.append("This report defines the current `query-critical token` protocol used for Phase A.")
    lines.append("")
    lines.append("- `TextVQA`: token-level protocol using dataset OCR tokens and ground-truth answer overlap.")
    lines.append("- `DocVQA`: current local mirror does not expose OCR tokens in the `DocVQA` config, so this phase falls back to answer-unit protocol and records the limitation explicitly.")
    lines.append("")

    for dataset_name in datasets:
        section = report[dataset_name]
        summary = section["summary"]
        lines.append(f"## {dataset_name}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(f"| total_samples | {summary['total_samples']} |")
        lines.append(f"| token_level_samples | {summary['token_level_samples']} |")
        lines.append(f"| token_level_ratio | {summary['token_level_ratio']:.4f} |")
        lines.append(f"| samples_with_key_tokens | {summary['samples_with_key_tokens']} |")
        lines.append(f"| samples_with_key_tokens_ratio | {summary['samples_with_key_tokens_ratio']:.4f} |")
        lines.append(f"| avg_key_tokens_per_sample | {summary['avg_key_tokens_per_sample']:.4f} |")
        lines.append("")
        lines.append("### Match Strategy Breakdown")
        lines.append("")
        lines.append("| Strategy | Count |")
        lines.append("|---|---:|")
        for key, value in summary["match_strategy_breakdown"].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        lines.append("### Note Breakdown")
        lines.append("")
        lines.append("| Note | Count |")
        lines.append("|---|---:|")
        if summary["note_breakdown"]:
            for key, value in summary["note_breakdown"].items():
                lines.append(f"| {key} | {value} |")
        else:
            lines.append("| none | 0 |")
        lines.append("")
        lines.append("### Representative Examples")
        lines.append("")
        for example in section["examples"]:
            lines.append(f"- `sample_id={example['sample_id']}`")
            lines.append(f"  - question: {example['question']}")
            lines.append(f"  - answers: {example['answers'][:3]}")
            lines.append(f"  - protocol_level: {example['protocol_level']}")
            lines.append(f"  - match_strategy: {example['match_strategy']}")
            lines.append(f"  - key_tokens: {example['key_tokens'][:12]}")
            if example.get("notes"):
                lines.append(f"  - notes: {example['notes']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    allowed = {"textvqa", "docvqa"}
    for dataset_name in datasets:
        if dataset_name not in allowed:
            raise ValueError(f"Unsupported dataset for Phase A.1: {dataset_name}")

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, dict] = {}
    for dataset_name in datasets:
        ds = VQADataset(
            dataset_name=dataset_name,
            split="eval",
            max_samples=args.max_samples,
            streaming=False,
            local_dir=args.local_data_dir,
        )
        records = [_sample_to_record(ds[idx], dataset_name, idx) for idx in range(len(ds))]
        report[dataset_name] = {
            "summary": summarize_key_token_annotations(records),
            "examples": _pick_examples(records, args.report_samples_per_dataset),
            "records": records,
        }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with out_md.open("w", encoding="utf-8") as f:
        f.write(_render_markdown(report, datasets))

    print(f"Saved JSON report to {out_json}")
    print(f"Saved Markdown report to {out_md}")


if __name__ == "__main__":
    main()
