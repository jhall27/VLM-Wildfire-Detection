"""Create fused masks from SAM masks and VLM refinement rules."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import cv2
import numpy as np
from PIL import Image

from eval_vlm import load_vlm_results
from vlm.refinement import (
    build_refinement_table,
    ensure_split_dirs,
    find_mask_split,
    make_blank_mask,
    make_eroded_mask,
    mirror_teacher_masks,
)


def validate_vlm_results(vlm_csv: Path) -> None:
    """Fail early on dry-run or incomplete VLM outputs."""
    raw_df = pd.read_csv(vlm_csv)
    required_cols = {"prompt_style", "region_name", "status", "response"}
    missing = required_cols - set(raw_df.columns)
    if missing:
        raise ValueError(f"VLM CSV is missing required columns: {sorted(missing)}")

    full_frame = raw_df[raw_df["region_name"] == "full_frame"].copy()
    if full_frame.empty:
        raise ValueError("No full-frame rows found in VLM CSV.")

    pending = full_frame[full_frame["status"] != "ok"]
    if not pending.empty:
        statuses = sorted(set(str(s) for s in pending["status"]))
        raise ValueError(
            "VLM CSV contains non-final rows for full-frame prompts "
            f"(status={statuses}). Run a real pilot first."
        )

    empty_responses = full_frame["response"].fillna("").astype(str).str.strip().eq("")
    if bool(empty_responses.any()):
        raise ValueError("VLM CSV contains empty full-frame responses.")


def write_fused_masks(
    refinement_df: pd.DataFrame,
    sam_dir: Path,
    fused_dir: Path,
    min_component_area: int,
) -> pd.DataFrame:
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
            note = write_component_filtered_mask(
                sam_mask_path,
                output_path,
                min_component_area=min_component_area,
                erode=True,
            )
            if "0 positive pixels remain" in note:
                note = "Component-filtered mask removed all pixels."
        elif action == "uncertain":
            # For mixed signals, keep only meaningful connected regions instead of
            # blindly copying every tiny SAM artifact.
            note = write_component_filtered_mask(
                sam_mask_path,
                output_path,
                min_component_area=min_component_area,
                erode=False,
            )
            if "0 positive pixels remain" in note:
                note = "Component-filtered mask removed all pixels."
        else:
            # Accept keeps the original teacher mask when smoke evidence is strongest.
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


def remove_small_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 0:
        return mask

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == label_idx] = 1
    return (cleaned * 255).astype(np.uint8)


def write_component_filtered_mask(source_mask: Path, output_path: Path, min_component_area: int, erode: bool = False) -> str:
    mask = np.array(Image.open(source_mask).convert("L"))
    cleaned = remove_small_components(mask, min_component_area)
    if erode and cleaned.any():
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.erode(cleaned, kernel, iterations=1)
    Image.fromarray(cleaned).save(output_path)
    remaining = int((cleaned > 0).sum())
    return f"Component-filtered mask written ({remaining} positive pixels remain)."


def print_refinement_summary(written_df: pd.DataFrame) -> None:
    print("\nRefinement action counts:")
    if written_df.empty:
        print("  No refinement rows were written.")
        return

    for action, count in written_df["action"].value_counts().items():
        print(f"  {action}: {count}")

    changed = written_df["action"].isin(["reject", "down_weight"]).sum()
    unchanged = written_df["action"].isin(["accept", "uncertain"]).sum()
    print(f"\nMasks changed from teacher labels: {changed}")
    print(f"Masks kept as teacher labels    : {unchanged}")


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
    parser.add_argument("--min-component-area", type=int, default=120)
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
        f"uncertain=[{args.uncertain_low},{args.uncertain_high}), "
        f"min_component>={args.min_component_area}"
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
    try:
        validate_vlm_results(vlm_csv)
    except ValueError as exc:
        print(f"Error: {exc}")
        print("Aborting to avoid generating misleading refinement outputs.")
        return

    vlm_df = load_vlm_results(vlm_csv)
    refinement_df = build_refinement_table(
        vlm_df,
        sam_dir=sam_dir,
        accept_threshold=args.accept_threshold,
        reject_threshold=args.reject_threshold,
        region_accept_threshold=args.region_accept_threshold,
        uncertain_low=args.uncertain_low,
        uncertain_high=args.uncertain_high,
    )
    written_df = write_fused_masks(
        refinement_df,
        sam_dir,
        fused_dir,
        min_component_area=args.min_component_area,
    )

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
    print_refinement_summary(written_df)


if __name__ == "__main__":
    main()
