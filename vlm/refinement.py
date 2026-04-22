"""Helpers for turning VLM outputs into training-ready refinement decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def ensure_split_dirs(root: Path) -> None:
    # Keep train/valid/test layout the same as the original baseline.
    for split in ["train", "valid", "test"]:
        (root / split).mkdir(parents=True, exist_ok=True)


def mirror_teacher_masks(source_root: Path, target_root: Path) -> Dict[str, int]:
    """
    Start from a full copy of the teacher labels so refined label folders stay
    training-ready even when we only have a few VLM decisions so far.
    """
    ensure_split_dirs(target_root)
    copied_per_split: Dict[str, int] = {}

    for split in ["train", "valid", "test"]:
        copied = 0
        for source_mask in (source_root / split).glob("*.png"):
            target_mask = target_root / split / source_mask.name
            target_mask.write_bytes(source_mask.read_bytes())
            copied += 1
        copied_per_split[split] = copied

    return copied_per_split


def find_mask_split(stem: str, sam_dir: Path) -> Tuple[Path | None, str | None]:
    # Search the baseline teacher masks so we can mirror the same split later.
    for split in ["train", "valid", "test"]:
        candidate = sam_dir / split / f"{stem}.png"
        if candidate.exists():
            return candidate, split
    return None, None


def make_blank_mask(source_mask: Path, output_path: Path) -> None:
    # Reuse the source mask size so the output stays training-ready.
    img = Image.open(source_mask)
    h, w = img.size[1], img.size[0]
    blank = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
    blank.save(output_path)


def make_eroded_mask(source_mask: Path, output_path: Path, kernel_size: int = 5) -> None:
    # A small erosion gives a simple "down-weighted" version of the teacher mask.
    mask = np.array(Image.open(source_mask).convert("L"))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    Image.fromarray(eroded).save(output_path)


def aggregate_full_frame_rows(vlm_df: pd.DataFrame) -> pd.DataFrame:
    # For the first pass, keep decisions simple and image-level using full-frame rows.
    full_frame = vlm_df[vlm_df["region_name"] == "full_frame"].copy()
    rows = []

    for image_path, group in full_frame.groupby("image_path"):
        record = {
            "image_path": image_path,
            "image_stem": Path(image_path).stem,
            "yes_no_smoke": None,
            "confidence_label": None,
            "confidence_score": None,
            "region_smoke": None,
            "region_stage": None,
            "region_confounder": None,
            "region_confidence": None,
        }

        for _, row in group.iterrows():
            style = row["prompt_style"]
            if style == "yes_no":
                record["yes_no_smoke"] = row.get("smoke_detected")
            elif style == "confidence_score":
                record["confidence_label"] = row.get("label")
                record["confidence_score"] = row.get("confidence")
            elif style == "region_reasoning":
                record["region_smoke"] = row.get("smoke_detected")
                record["region_stage"] = row.get("smoke_stage")
                record["region_confounder"] = row.get("confounder")
                record["region_confidence"] = row.get("confidence")

        rows.append(record)

    return pd.DataFrame(rows)


def decide_refinement_action(
    row: pd.Series,
    accept_threshold: int = 70,
    reject_threshold: int = 40,
    region_accept_threshold: int = 60,
    uncertain_low: int = 40,
    uncertain_high: int = 70,
) -> Tuple[str, str]:
    # These rules are intentionally simple for the first VLM-guided pass.
    yes_no_smoke = row.get("yes_no_smoke")
    confidence_label = str(row.get("confidence_label") or "").lower()
    confidence_score = row.get("confidence_score")
    region_smoke = row.get("region_smoke")
    region_stage = str(row.get("region_stage") or "").lower()
    region_confounder = str(row.get("region_confounder") or "").lower()
    region_confidence = row.get("region_confidence")

    if (
        yes_no_smoke == 0
        and confidence_label in {"background", "cloud_or_fog", "uncertain", ""}
        and (confidence_score is None or confidence_score >= reject_threshold)
        and region_smoke == 0
    ):
        return "reject", "All full-frame prompts leaned away from smoke with confident negative evidence."

    if (
        yes_no_smoke == 1
        and confidence_label == "smoke"
        and (confidence_score or 0) >= accept_threshold
        and region_smoke == 1
        and (region_confidence or 0) >= region_accept_threshold
    ):
        return "accept", "All full-frame prompts strongly supported smoke."

    if (
        confidence_label == "smoke"
        and uncertain_low <= (confidence_score or 0) < uncertain_high
    ) or (
        region_smoke == 1
        and (region_stage == "early" or region_confounder in {"haze", "cloud", "fog"})
    ):
        return "down_weight", "Smoke was detected, but confidence/confounders suggest caution."

    # Mixed signals are kept separate so we can inspect them later.
    return "uncertain", "Signals were mixed, so the original teacher mask is kept."


def build_refinement_table(
    vlm_df: pd.DataFrame,
    accept_threshold: int = 70,
    reject_threshold: int = 40,
    region_accept_threshold: int = 60,
    uncertain_low: int = 40,
    uncertain_high: int = 70,
) -> pd.DataFrame:
    aggregated = aggregate_full_frame_rows(vlm_df)
    actions = aggregated.apply(
        decide_refinement_action,
        axis=1,
        result_type="expand",
        accept_threshold=accept_threshold,
        reject_threshold=reject_threshold,
        region_accept_threshold=region_accept_threshold,
        uncertain_low=uncertain_low,
        uncertain_high=uncertain_high,
    )
    aggregated["action"] = actions[0]
    aggregated["decision_reason"] = actions[1]
    aggregated["accept_threshold"] = accept_threshold
    aggregated["reject_threshold"] = reject_threshold
    aggregated["region_accept_threshold"] = region_accept_threshold
    aggregated["uncertain_low"] = uncertain_low
    aggregated["uncertain_high"] = uncertain_high
    return aggregated
