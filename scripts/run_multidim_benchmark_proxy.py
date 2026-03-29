#!/usr/bin/env python3
"""Phase 2.4: multi-dimension benchmark proxy evaluation."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def create_demo_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (640, 384), color=(238, 244, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        (48, 52, 300, 220), fill=(83, 141, 213), outline=(34, 75, 132), width=4
    )
    draw.ellipse(
        (360, 78, 600, 318), fill=(245, 195, 88), outline=(131, 95, 20), width=4
    )
    draw.text((80, 260), "QACR", fill=(20, 20, 20))
    img.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B")
    parser.add_argument("--image", default="outputs/demo_phase01.png")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--save-json",
        default="outputs/phase24_multidim_proxy_results.json",
    )
    return parser.parse_args()


def build_proxy_samples() -> list[dict[str, Any]]:
    return [
        {
            "dataset": "VQAv2",
            "question": "图中主要包含哪两种几何形状？",
            "kind": "keywords_all",
            "keywords": ["矩形", "圆"],
        },
        {
            "dataset": "VQAv2",
            "question": "图中蓝色区域是什么形状？",
            "kind": "keywords_any",
            "keywords": ["矩形", "长方形"],
        },
        {
            "dataset": "GQA",
            "question": "黄色圆形在蓝色矩形的左边还是右边？",
            "kind": "keywords_any",
            "keywords": ["右", "右边"],
        },
        {
            "dataset": "GQA",
            "question": "文字 QACR 在几何图形的上方还是下方？",
            "kind": "keywords_any",
            "keywords": ["下", "下方"],
        },
        {
            "dataset": "POPE",
            "question": "图里有绿色三角形吗？请只回答是或否。",
            "kind": "yesno",
            "expected_yes": False,
        },
        {
            "dataset": "POPE",
            "question": "图里有蓝色矩形吗？请只回答是或否。",
            "kind": "yesno",
            "expected_yes": True,
        },
        {
            "dataset": "TextVQA",
            "question": "图中的英文文本是什么？",
            "kind": "keywords_any",
            "keywords": ["qacr"],
        },
        {
            "dataset": "TextVQA",
            "question": "请读取图中可见文字内容。",
            "kind": "keywords_any",
            "keywords": ["qacr"],
        },
        {
            "dataset": "DocVQA",
            "question": "如果把这张图当作文档页面，页面上唯一明显的词是什么？",
            "kind": "keywords_any",
            "keywords": ["qacr"],
        },
        {
            "dataset": "DocVQA",
            "question": "文档中是否出现 QACR 这个词？请回答是或否。",
            "kind": "yesno",
            "expected_yes": True,
        },
        {
            "dataset": "MMBench",
            "question": "图中有几个主要几何图形（不含文字）？",
            "kind": "keywords_any",
            "keywords": ["2", "两个", "二个", "两"],
        },
        {
            "dataset": "MMMU",
            "question": "若将蓝色矩形记为A、黄色圆形记为B，哪个目标更靠右？",
            "kind": "keywords_any",
            "keywords": ["b", "B", "圆", "黄色"],
        },
    ]


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def run_infer(
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    image_path: Path,
    query: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": query},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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
        k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()
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
    t = text.strip().lower()
    yes_markers = ["是", "有", "yes", "存在", "对"]
    no_markers = ["否", "没有", "无", "不", "no", "不存在"]
    yes_hit = any(x in t for x in yes_markers)
    no_hit = any(x in t for x in no_markers)
    if yes_hit and not no_hit:
        return True
    if no_hit and not yes_hit:
        return False
    return None


def evaluate(sample: dict[str, Any], response: str) -> bool:
    r = response.lower()
    kind = sample["kind"]
    if kind == "keywords_any":
        return any(k.lower() in r for k in sample["keywords"])
    if kind == "keywords_all":
        return all(k.lower() in r for k in sample["keywords"])
    if kind == "yesno":
        pred = judge_yes_no(response)
        if pred is None:
            return False
        return bool(pred) == bool(sample["expected_yes"])
    raise ValueError(f"Unknown kind: {kind}")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    out_json = Path(args.save_json)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not image_path.exists():
        create_demo_image(image_path)

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

    samples = build_proxy_samples()
    results = []
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for sample in samples:
        response = run_infer(
            processor=processor,
            model=model,
            image_path=image_path,
            query=sample["question"],
            max_new_tokens=args.max_new_tokens,
        )
        ok = evaluate(sample, response)
        ds = sample["dataset"]
        stats[ds]["total"] += 1
        stats[ds]["correct"] += int(ok)
        results.append(
            {
                "dataset": ds,
                "question": sample["question"],
                "response": response,
                "correct": ok,
                "judge_kind": sample["kind"],
            }
        )

    per_dataset = {}
    macro_sum = 0.0
    for ds in sorted(stats.keys()):
        c = stats[ds]["correct"]
        t = stats[ds]["total"]
        acc = c / max(t, 1)
        per_dataset[ds] = {"correct": c, "total": t, "accuracy": acc}
        macro_sum += acc
    macro_acc = macro_sum / max(len(per_dataset), 1)
    micro_correct = sum(v["correct"] for v in stats.values())
    micro_total = sum(v["total"] for v in stats.values())
    micro_acc = micro_correct / max(micro_total, 1)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "image": str(image_path),
                "proxy_samples": len(samples),
                "per_dataset": per_dataset,
                "macro_accuracy": macro_acc,
                "micro_accuracy": micro_acc,
                "details": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("===== Multi-Dimension Benchmark Proxy Eval (Task 2.4) =====")
    print(f"model: {model_path}")
    print(f"image: {image_path}")
    print(f"proxy_samples: {len(samples)}")
    print("dataset | correct | total | accuracy")
    for ds in sorted(per_dataset.keys()):
        item = per_dataset[ds]
        print(
            f"{ds} | {item['correct']} | {item['total']} | {item['accuracy']:.6f}"
        )
    print(f"macro_accuracy: {macro_acc:.6f}")
    print(f"micro_accuracy: {micro_acc:.6f}")
    print(f"saved_json: {out_json}")
    print("status: proxy multi-benchmark evaluation succeeded")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

