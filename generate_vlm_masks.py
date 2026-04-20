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
import re

import numpy as np
import pandas as pd
from PIL import Image
from vlm.refinement import ensure_split_dirs, find_mask_split, mirror_teacher_masks


def _parse_yes_no_response(response: str) -> int | None:
    """Parse strict yes/no output; return None when unparseable."""
    text = str(response or "").strip().lower()
    if not text:
        return None

    # Only accept an explicit yes/no token to avoid accidental matches.
    match = re.match(r"^(yes|no)\b", text)
    if not match:
        return None
    return 1 if match.group(1) == "yes" else 0


def get_yes_no_decisions(vlm_csv: Path) -> dict:
    """
    Read qwen_pilot_results.csv and return a dict of
    {image_stem: smoke_detected (0 or 1)} 

    region == full frame
    """
    df = pd.read_csv(vlm_csv)
    required_cols = {"image_path", "prompt_style", "region_name", "response", "status"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"VLM CSV is missing required columns: {sorted(missing)}")

    # Start with the most conservative setup: only use the full-frame yes/no answer.
    filtered = df[
        (df["prompt_style"] == "yes_no") &
        (df["region_name"] == "full_frame")
    ].copy()
    if filtered.empty:
        raise ValueError("No full-frame yes/no rows found in VLM CSV.")

    pending_rows = filtered[filtered["status"] != "ok"]
    if not pending_rows.empty:
        bad_states = sorted(set(str(s) for s in pending_rows["status"]))
        raise ValueError(
            "VLM CSV contains non-final rows for yes/no full-frame prompts "
            f"(status={bad_states}). Run a real pilot first "
            "(`python3 vlm/run_qwen_pilot.py --mode local ...`)."
        )

    decisions = {}
    invalid_rows = []
    conflicting_stems = []
    for _, row in filtered.iterrows():
        img_path = Path(row["image_path"])
        stem = img_path.stem  # e.g. "0_2777"
        smoke = _parse_yes_no_response(row["response"])
        if smoke is None:
            invalid_rows.append(stem)
            continue
        if stem in decisions and decisions[stem] != smoke:
            conflicting_stems.append(stem)
            continue
        decisions[stem] = smoke

    if invalid_rows:
        raise ValueError(
            "Some yes/no responses were empty or not parseable as strict yes/no. "
            f"Example stems: {sorted(set(invalid_rows))[:5]}"
        )
    if conflicting_stems:
        raise ValueError(
            "Conflicting yes/no decisions found for image stems: "
            f"{sorted(set(conflicting_stems))[:10]}"
        )
    if not decisions:
        raise ValueError("No valid yes/no decisions were parsed from VLM CSV.")

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

    try:
        decisions = get_yes_no_decisions(vlm_csv)
    except ValueError as exc:
        print(f"Error: {exc}")
        print("Aborting to avoid writing incorrect blank masks.")
        return
    print(f"VLM decisions loaded: {len(decisions)} image(s)")
    for stem, smoke in decisions.items():
        print(f"  {stem} -> {'smoke' if smoke else 'no smoke'}")
    print()

    apply_vlm_decisions(decisions, sam_dir, vlm_dir)



if __name__ == "__main__":
    main()
