# Reads the VLM results and writes simple VLM-filtered masks.
"""
Uses the yes_no full_frame VLM decision (most stable prompt style):
  - VLM said "no"  → save a blank (all-zero) mask to vlm_masks/
  - VLM said "yes" → copy the SAM mask unchanged to vlm_masks/

Usage:
    python generate_vlm_masks.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from vlm.refinement import ensure_split_dirs, find_mask_split, mirror_teacher_masks


def get_yes_no_decisions(vlm_csv: Path) -> dict:
    """
    Read qwen_pilot_results.csv and return a dict of
    {image_stem: smoke_detected (0 or 1)} 

    region == full frame
    """
    df = pd.read_csv(vlm_csv)

    # Start with the most conservative setup: only use the full-frame yes/no answer.
    filtered = df[
        (df["prompt_style"] == "yes_no") &
        (df["region_name"] == "full_frame")
    ].copy()

    decisions = {}
    for _, row in filtered.iterrows():
        img_path = Path(row["image_path"])
        stem = img_path.stem  # e.g. "0_2777"
        response = str(row["response"]).strip().lower()
        smoke = 1 if "yes" in response else 0
        decisions[stem] = smoke

    return decisions


def apply_vlm_decisions(
    decisions: dict,
    sam_dir: Path,
    vlm_dir: Path,
) -> None:
    """
    For each image with a VLM decision, overwrite the mask in vlm_dir:
      - smoke == 0 (VLM said no) → blank mask
      - smoke == 1 (VLM said yes) → copy SAM mask unchanged
    """
    changed = 0
    skipped = 0

    for stem, smoke in decisions.items():
        # Find the matching teacher mask first so we keep the original split.
        sam_mask_path, split_found = find_mask_split(stem, sam_dir)

        if sam_mask_path is None:
            print(f"  [SKIP] No SAM mask found for '{stem}' in any split")
            skipped += 1
            continue

        vlm_mask_path = vlm_dir / split_found / f"{stem}.png"

        if smoke == 0:
            action = "blank mask (VLM said no smoke)"
            img = Image.open(sam_mask_path)
            h, w = img.size[1], img.size[0]
            # Blank mask means the VLM rejected this candidate as smoke.
            blank = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
            blank.save(vlm_mask_path)
        else:
            action = "SAM mask kept (VLM confirmed smoke)"
            shutil.copy(sam_mask_path, vlm_mask_path)

        print(f"  [{split_found}] {stem}.png -> {action}")
        changed += 1

    print(f"\nDone. {changed} mask(s) updated, {skipped} skipped.")


def main():
    vlm_csv = Path("vlm_outputs/qwen_pilot_results.csv")
    sam_dir = Path("data/sam_masks")
    vlm_dir = Path("data/vlm_masks")

    ensure_split_dirs(vlm_dir)

    if not vlm_csv.exists():
        print(f"Error: VLM results not found: {vlm_csv}")
        return

    print("=" * 60)
    print("GENERATE VLM-REFINED MASKS")
    print("=" * 60)
    print(f"VLM results : {vlm_csv}")
    print(f"SAM masks   : {sam_dir}")
    print(f"VLM masks   : {vlm_dir}")
    print()

    # Builds a complete drop-in label folder first, then overwrite only the
    # masks that received a VLM decision.
    copied_per_split = mirror_teacher_masks(sam_dir, vlm_dir)
    print("Seeded vlm_masks from teacher masks:")
    for split, count in copied_per_split.items():
        print(f"  {split}: {count} copied")
    print()

    decisions = get_yes_no_decisions(vlm_csv)
    print(f"VLM decisions loaded: {len(decisions)} image(s)")
    for stem, smoke in decisions.items():
        print(f"  {stem} -> {'smoke' if smoke else 'no smoke'}")
    print()

    apply_vlm_decisions(decisions, sam_dir, vlm_dir)



if __name__ == "__main__":
    main()
