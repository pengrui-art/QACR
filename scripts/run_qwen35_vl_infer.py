#!/usr/bin/env python3
"""Run local Qwen3.5-VL image+text inference as Phase 0.1 baseline smoke test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    parser.add_argument("--model", default="Model/Qwen35-08B", help="Local model path")
    parser.add_argument(
        "--image", default="outputs/demo_phase01.png", help="Input image path"
    )
    parser.add_argument(
        "--query",
        default="请用一句话描述这张图中主要的形状和文字。",
        help="Text query",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)

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
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": args.query},
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
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("===== Qwen3.5-VL Inference Smoke Test =====")
    print(f"model: {model_path}")
    print(f"device: {model_device}")
    print(f"dtype: {dtype}")
    print(f"image: {image_path}")
    print(f"query: {args.query}")
    print(f"response: {response}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
