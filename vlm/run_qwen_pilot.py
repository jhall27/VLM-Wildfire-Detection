"""Run a simple Qwen pilot or create a dry-run results sheet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from PIL import Image

try:
    from vlm.prompt_templates import PROMPT_TEMPLATES, get_prompt
except ModuleNotFoundError:
    # This keeps the script usable with "python3 vlm/run_qwen_pilot.py".
    from prompt_templates import PROMPT_TEMPLATES, get_prompt


def crop_region(image: Image.Image, x0: float, y0: float, x1: float, y1: float) -> Image.Image:
    width, height = image.size
    left = int(width * x0)
    top = int(height * y0)
    right = int(width * x1)
    bottom = int(height * y1)
    return image.crop((left, top, right, bottom))


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def iter_styles(requested_style: str) -> Iterable[str]:
    if requested_style == "all":
        return PROMPT_TEMPLATES.keys()
    return [requested_style]


def run_dry(
    manifest_rows: list[dict[str, str]],
    output_path: Path,
    style: str,
    max_samples: int | None,
) -> None:
    rows = []
    for manifest_row in manifest_rows[:max_samples]:
        for prompt_style in iter_styles(style):
            rows.append(
                {
                    "image_path": manifest_row["image_path"],
                    "case_type": manifest_row["case_type"],
                    "region_name": manifest_row["region_name"],
                    "prompt_style": prompt_style,
                    "prompt_text": get_prompt(prompt_style),
                    "model_name": "qwen2.5-vl-placeholder",
                    "status": "pending_model_run",
                    "response": "",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "case_type",
                "region_name",
                "prompt_style",
                "prompt_text",
                "model_name",
                "status",
                "response",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote dry-run template with {len(rows)} rows to {output_path}")


def run_local_qwen(
    manifest_rows: list[dict[str, str]],
    output_path: Path,
    style: str,
    max_samples: int | None,
    model_name: str,
    device: str,
) -> None:
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:
        raise SystemExit(
            "Real Qwen runs need transformers installed. "
            "Use --mode dry-run for now, or install transformers first."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")
    model = model.to(device)

    rows = []
    for manifest_row in manifest_rows[:max_samples]:
        image = Image.open(manifest_row["image_path"]).convert("RGB")
        cropped = crop_region(
            image,
            float(manifest_row["x0"]),
            float(manifest_row["y0"]),
            float(manifest_row["x1"]),
            float(manifest_row["y1"]),
        )

        for prompt_style in iter_styles(style):
            prompt_text = get_prompt(prompt_style)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": cropped},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[cropped], padding=True, return_tensors="pt")
            inputs = inputs.to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_text = processor.batch_decode(
                generated_ids[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            rows.append(
                {
                    "image_path": manifest_row["image_path"],
                    "case_type": manifest_row["case_type"],
                    "region_name": manifest_row["region_name"],
                    "prompt_style": prompt_style,
                    "prompt_text": prompt_text,
                    "model_name": model_name,
                    "status": "ok",
                    "response": generated_text.strip(),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "case_type",
                "region_name",
                "prompt_style",
                "prompt_text",
                "model_name",
                "status",
                "response",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Qwen results to {output_path}")


def write_run_config(args: argparse.Namespace, output_path: Path) -> None:
    config_path = output_path.with_suffix(".json")
    config_path.write_text(json.dumps(vars(args), indent=2))
    print(f"Wrote run config to {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or stage a Qwen VLM pilot on hard wildfire cases.")
    parser.add_argument("--manifest", default="vlm_outputs/hard_case_manifest.csv")
    parser.add_argument("--output", default="vlm_outputs/qwen_pilot_results.csv")
    parser.add_argument("--mode", choices=["dry-run", "local"], default="dry-run")
    parser.add_argument("--prompt-style", choices=["yes_no", "confidence_score", "region_reasoning", "all"], default="all")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    manifest_rows = load_manifest(manifest_path)
    write_run_config(args, output_path)

    if args.mode == "dry-run":
        run_dry(manifest_rows, output_path, args.prompt_style, args.max_samples)
    else:
        run_local_qwen(
            manifest_rows,
            output_path,
            args.prompt_style,
            args.max_samples,
            args.model_name,
            args.device,
        )


if __name__ == "__main__":
    main()
