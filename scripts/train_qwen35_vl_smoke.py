#!/usr/bin/env python3
"""One-step fine-tuning smoke test for local Qwen3.5-VL baseline.

This verifies forward + backward + optimizer step can run end-to-end.
"""

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
    img = Image.new("RGB", (640, 384), color=(250, 247, 241))
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        (52, 60, 320, 220), fill=(104, 170, 120), outline=(44, 94, 56), width=4
    )
    draw.ellipse(
        (360, 70, 608, 320), fill=(210, 130, 92), outline=(117, 55, 29), width=4
    )
    draw.text((86, 262), "QACR", fill=(20, 20, 20))
    img.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Model/Qwen35-08B", help="Local model path")
    parser.add_argument(
        "--image", default="outputs/demo_phase01_train.png", help="Input image path"
    )
    parser.add_argument("--lr", type=float, default=1e-5)
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
    model.train()

    # Keep trainable params minimal for a fast smoke test.
    for p in model.parameters():
        p.requires_grad = False

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Expected lm_head in model, but it was not found.")

    for p in lm_head.parameters():
        p.requires_grad = True

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.resolve())},
                {"type": "text", "text": "请描述图里的主要形状和文字。"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "图中有一个蓝绿色矩形、一个橙色圆形区域，并且包含文字QACR。",
                }
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
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

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    optimizer.zero_grad(set_to_none=True)

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print("===== Qwen3.5-VL Fine-tuning Smoke Test =====")
    print(f"model: {model_path}")
    print(f"device: {model_device}")
    print(f"dtype: {dtype}")
    print(f"image: {image_path}")
    print(
        f"trainable_parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    print(f"loss: {float(loss.detach().cpu()):.6f}")
    print("status: one-step forward/backward/optimizer succeeded")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
