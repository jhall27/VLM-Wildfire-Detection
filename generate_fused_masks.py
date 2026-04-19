"""Create fused masks from SAM masks and VLM refinement rules."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval_vlm import load_vlm_results
from vlm.refinement import (
    build_refinement_table,
    ensure_split_dirs,
    find_mask_split,
    make_blank_mask,
    make_eroded_mask,
    mirror_teacher_masks,
)


def write_fused_masks(refinement_df: pd.DataFrame, sam_dir: Path, fused_dir: Path) -> pd.DataFrame:
    ensure_split_dirs(fused_dir)

    written_rows = []
    for _, row in refinement_df.iterrows():
        stem = row["image_stem"]
        action = row["action"]
        sam_mask_path, split_found = find_mask_split(stem, sam_dir)

        if sam_mask_path is None:
            written_rows.append(
                {
                    **row.to_dict(),
                    "split": None,
                    "mask_written": False,
                    "mask_path": None,
                    "write_note": "No matching SAM mask found in train/valid/test.",
                }
            )
            continue

        output_path = fused_dir / split_found / f"{stem}.png"
        if action == "reject":
            # Reject = VLM thinks this should not be treated as smoke.
            make_blank_mask(sam_mask_path, output_path)
            note = "Blank mask written."
        elif action == "down_weight":
            # Down-weight = keep the region, but shrink it to be more conservative.
            make_eroded_mask(sam_mask_path, output_path)
            note = "Eroded mask written."
        else:
            # Accept or uncertain both keep the original teacher mask for now.
            output_path.write_bytes(sam_mask_path.read_bytes())
            note = "Original SAM mask copied."

        written_rows.append(
            {
                **row.to_dict(),
                "split": split_found,
                "mask_written": True,
                "mask_path": str(output_path),
                "write_note": note,
            }
        )

    return pd.DataFrame(written_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fused masks from VLM refinement decisions.")
    parser.add_argument("--vlm-results", default="vlm_outputs/qwen_pilot_results.csv")
    parser.add_argument("--sam-dir", default="data/sam_masks")
    parser.add_argument("--fused-dir", default="data/fused_masks")
    parser.add_argument("--output-csv", default="vlm_outputs/fused_refinement_decisions.csv")
    parser.add_argument("--accept-threshold", type=int, default=70)
    parser.add_argument("--reject-threshold", type=int, default=40)
    parser.add_argument("--region-accept-threshold", type=int, default=60)
    parser.add_argument("--uncertain-low", type=int, default=40)
    parser.add_argument("--uncertain-high", type=int, default=70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vlm_csv = Path(args.vlm_results)
    sam_dir = Path(args.sam_dir)
    fused_dir = Path(args.fused_dir)
    refinement_csv = Path(args.output_csv)

    if not vlm_csv.exists():
        print(f"Error: VLM results not found: {vlm_csv}")
        return

    print("=" * 60)
    print("GENERATE FUSED MASKS")
    print("=" * 60)
    print(f"VLM results : {vlm_csv}")
    print(f"SAM masks   : {sam_dir}")
    print(f"Fused masks : {fused_dir}")
    print(
        "Thresholds  : "
        f"accept>={args.accept_threshold}, "
        f"reject<={args.reject_threshold}, "
        f"region_accept>={args.region_accept_threshold}, "
        f"uncertain=[{args.uncertain_low},{args.uncertain_high})"
    )
    print()

    # Start from a complete mirror of the teacher masks so the fused label set
    # can be used directly during training, even before we have many VLM runs.
    copied_per_split = mirror_teacher_masks(sam_dir, fused_dir)
    print("Seeded fused_masks from teacher masks:")
    for split, count in copied_per_split.items():
        print(f"  {split}: {count} copied")
    print()

    # Parse the saved Qwen outputs first, then turn them into simple actions.
    vlm_df = load_vlm_results(vlm_csv)
    refinement_df = build_refinement_table(
        vlm_df,
        accept_threshold=args.accept_threshold,
        reject_threshold=args.reject_threshold,
        region_accept_threshold=args.region_accept_threshold,
        uncertain_low=args.uncertain_low,
        uncertain_high=args.uncertain_high,
    )
    written_df = write_fused_masks(refinement_df, sam_dir, fused_dir)

    written_df.to_csv(refinement_csv, index=False)
    print(f"Saved refinement decisions: {refinement_csv}")
    print()
    print(written_df[[
        "image_stem",
        "action",
        "decision_reason",
        "split",
        "mask_written",
        "write_note",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
