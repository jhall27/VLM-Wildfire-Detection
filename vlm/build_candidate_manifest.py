"""Build a VLM manifest from teacher masks instead of only hand-picked hard cases."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the tight bounding box around non-zero pixels."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_box(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    """Add some context around the teacher mask so the VLM sees nearby scene cues."""
    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    pad_x = int(box_w * margin_ratio)
    pad_y = int(box_h * margin_ratio)
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(image_width, x1 + pad_x),
        min(image_height, y1 + pad_y),
    )


def to_norm_box(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (
        x0 / image_width,
        y0 / image_height,
        x1 / image_width,
        y1 / image_height,
    )


def build_candidate_rows(
    data_dir: Path,
    split: str,
    margin_ratio: float,
    include_full_frame: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    image_dir = data_dir / "images" / split
    mask_dir = data_dir / "sam_masks" / split

    for mask_path in sorted(mask_dir.glob("*.png")):
        image_path = image_dir / f"{mask_path.stem}.jpeg"
        if not image_path.exists():
            continue

        mask = np.array(Image.open(mask_path).convert("L"))
        bbox = mask_bbox(mask)
        if bbox is None:
            continue

        height, width = mask.shape
        candidate_box = expand_box(bbox, width, height, margin_ratio)
        norm_x0, norm_y0, norm_x1, norm_y1 = to_norm_box(candidate_box, width, height)

        rows.append(
            {
                "image_path": str(image_path),
                "split": split,
                "case_type": "teacher_candidate",
                "region_name": "teacher_box_context",
                "candidate_kind": "teacher_box_context",
                "source_mask": str(mask_path),
                "pixel_area": int((mask > 0).sum()),
                "x0": round(norm_x0, 6),
                "y0": round(norm_y0, 6),
                "x1": round(norm_x1, 6),
                "y1": round(norm_y1, 6),
            }
        )

        if include_full_frame:
            rows.append(
                {
                    "image_path": str(image_path),
                    "split": split,
                    "case_type": "teacher_candidate",
                    "region_name": "full_frame",
                    "candidate_kind": "full_frame",
                    "source_mask": str(mask_path),
                    "pixel_area": int((mask > 0).sum()),
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 1.0,
                    "y1": 1.0,
                }
            )

    return rows


def build_manifest(
    data_dir: Path,
    output_path: Path,
    splits: list[str],
    margin_ratio: float,
    include_full_frame: bool,
) -> int:
    rows: list[dict[str, object]] = []
    for split in splits:
        rows.extend(build_candidate_rows(data_dir, split, margin_ratio, include_full_frame))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "split",
                "case_type",
                "region_name",
                "candidate_kind",
                "source_mask",
                "pixel_area",
                "x0",
                "y0",
                "x1",
                "y1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a VLM manifest from teacher masks.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="vlm_outputs/teacher_candidate_manifest.csv")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--margin-ratio", type=float, default=0.2)
    parser.add_argument("--no-full-frame", action="store_true")
    args = parser.parse_args()

    count = build_manifest(
        Path(args.data_dir),
        Path(args.output),
        args.splits,
        args.margin_ratio,
        include_full_frame=not args.no_full_frame,
    )
    print(f"Wrote {count} teacher-candidate rows to {args.output}")


if __name__ == "__main__":
    main()
