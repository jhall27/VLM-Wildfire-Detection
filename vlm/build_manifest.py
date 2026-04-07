"""Build a small manifest for full-frame and region-level VLM checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_REGIONS = [
    ("full_frame", 0.0, 0.0, 1.0, 1.0),
    ("grid_top_left", 0.0, 0.0, 0.5, 0.5),
    ("grid_top_right", 0.5, 0.0, 1.0, 0.5),
    ("grid_bottom_left", 0.0, 0.5, 0.5, 1.0),
    ("grid_bottom_right", 0.5, 0.5, 1.0, 1.0),
]


def build_manifest(hard_case_dir: Path, output_path: Path) -> int:
    rows = []
    for image_path in sorted(hard_case_dir.rglob("*.jpeg")):
        case_type = image_path.parent.name
        for region_name, x0, y0, x1, y1 in DEFAULT_REGIONS:
            rows.append(
                {
                    "image_path": str(image_path),
                    "case_type": case_type,
                    "region_name": region_name,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "case_type", "region_name", "x0", "y0", "x1", "y1"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hard-case manifest for VLM pilot runs.")
    parser.add_argument("--hard-case-dir", default="hard_cases")
    parser.add_argument("--output", default="vlm_outputs/hard_case_manifest.csv")
    args = parser.parse_args()

    count = build_manifest(Path(args.hard_case_dir), Path(args.output))
    print(f"Wrote {count} manifest rows to {args.output}")


if __name__ == "__main__":
    main()
