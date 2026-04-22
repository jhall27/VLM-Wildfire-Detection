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


def get_mask_area(stem: str, sam_dir: Path) -> int:
    source_mask, _ = find_mask_split(stem, sam_dir)
    if source_mask is None:
        return 0
    mask = np.array(Image.open(source_mask).convert("L")) > 0
    return int(mask.sum())


def aggregate_region_rows(vlm_df: pd.DataFrame, region_name: str, prefix: str) -> pd.DataFrame:
    region_rows = vlm_df[vlm_df["region_name"] == region_name].copy()
    rows = []

    for image_path, group in region_rows.groupby("image_path"):
        record = {
            "image_path": image_path,
            "image_stem": Path(image_path).stem,
            f"{prefix}_yes_no_smoke": None,
            f"{prefix}_confidence_label": None,
            f"{prefix}_confidence_score": None,
            f"{prefix}_region_smoke": None,
            f"{prefix}_region_stage": None,
            f"{prefix}_region_confounder": None,
            f"{prefix}_region_confidence": None,
        }

        for _, row in group.iterrows():
            style = row["prompt_style"]
            if style == "yes_no":
                record[f"{prefix}_yes_no_smoke"] = row.get("smoke_detected")
            elif style == "confidence_score":
                record[f"{prefix}_confidence_label"] = row.get("label")
                record[f"{prefix}_confidence_score"] = row.get("confidence")
            elif style == "region_reasoning":
                record[f"{prefix}_region_smoke"] = row.get("smoke_detected")
                record[f"{prefix}_region_stage"] = row.get("smoke_stage")
                record[f"{prefix}_region_confounder"] = row.get("confounder")
                record[f"{prefix}_region_confidence"] = row.get("confidence")

        rows.append(record)

    return pd.DataFrame(rows)


def decide_refinement_action(
    row: pd.Series,
    accept_threshold: int = 70,
    reject_threshold: int = 40,
    region_accept_threshold: int = 60,
    uncertain_low: int = 40,
    uncertain_high: int = 70,
    tiny_mask_threshold: int = 120,
    down_weight_min_area: int = 256,
    refinement_strategy: str = "hybrid",
) -> Tuple[str, str]:
    # Default strategy is hybrid: local teacher-region evidence first, with
    # full-frame UAV context used as a guardrail against crop-only mistakes.
    yes_no_smoke = row.get("full_yes_no_smoke")
    confidence_label = str(row.get("full_confidence_label") or "").lower()
    confidence_score = row.get("full_confidence_score")
    region_smoke = row.get("full_region_smoke")
    region_stage = str(row.get("full_region_stage") or "").lower()
    region_confounder = str(row.get("full_region_confounder") or "").lower()
    region_confidence = row.get("full_region_confidence")

    local_yes_no_smoke = row.get("crop_yes_no_smoke")
    local_confidence_label = str(row.get("crop_confidence_label") or "").lower()
    local_confidence_score = row.get("crop_confidence_score")
    local_region_smoke = row.get("crop_region_smoke")
    local_region_stage = str(row.get("crop_region_stage") or "").lower()
    local_region_confounder = str(row.get("crop_region_confounder") or "").lower()
    local_region_confidence = row.get("crop_region_confidence")
    mask_area = int(row.get("mask_area") or 0)
    negative_labels = {"background", "cloud_or_fog", "uncertain", ""}
    atmospheric_confounders = {"haze", "cloud", "fog", "background", "none", ""}
    negative_full_frame = yes_no_smoke == 0 and confidence_label in negative_labels
    negative_local = local_yes_no_smoke == 0 and local_confidence_label in negative_labels
    confident_negative = confidence_score is None or confidence_score >= reject_threshold
    confident_local_negative = local_confidence_score is None or local_confidence_score >= reject_threshold
    strong_positive = (
        yes_no_smoke == 1
        and confidence_label == "smoke"
        and (confidence_score or 0) >= accept_threshold
        and region_smoke == 1
        and (region_confidence or 0) >= region_accept_threshold
    )
    strong_local_positive = (
        local_yes_no_smoke == 1
        and local_confidence_label == "smoke"
        and (local_confidence_score or 0) >= accept_threshold
        and local_region_smoke == 1
        and (local_region_confidence or 0) >= region_accept_threshold
    )
    full_frame_supportive = (
        yes_no_smoke == 1
        or confidence_label == "smoke"
        or region_smoke == 1
    )
    full_frame_strong_positive = strong_positive
    cautious_positive = (
        local_yes_no_smoke == 1
        and local_confidence_label == "smoke"
        and local_region_smoke == 1
    )

    if refinement_strategy == "full_frame_only":
        if mask_area > 0 and mask_area < tiny_mask_threshold and not strong_positive:
            return "reject", "Teacher mask is too small and ambiguous to trust as supervision."

        if (
            negative_full_frame
            and confident_negative
            and region_smoke == 0
        ):
            return "reject", "All full-frame prompts leaned away from smoke with confident negative evidence."

        if (
            negative_full_frame
            and confident_negative
            and region_smoke == 1
            and region_confounder in atmospheric_confounders
        ):
            return "reject", "Full-frame prompts rejected smoke and region reasoning only suggested an atmospheric confounder."

        if strong_positive:
            return "accept", "All full-frame prompts strongly supported smoke."

        if (
            yes_no_smoke == 1
            and confidence_label == "smoke"
            and mask_area >= down_weight_min_area
            and (
                uncertain_low <= (confidence_score or 0) < uncertain_high
                or region_stage == "early"
                or region_confounder in {"haze", "cloud", "fog"}
            )
        ):
            return "down_weight", "Full-frame prompts detected smoke, but confidence/confounders suggest caution."

        return "uncertain", "Full-frame signals were mixed, so the original teacher mask is kept."

    if mask_area > 0 and mask_area < tiny_mask_threshold and not strong_local_positive and not strong_positive:
        return "reject", "Teacher mask is too small and ambiguous to trust as supervision."

    if (
        negative_local
        and confident_local_negative
        and local_region_smoke == 0
    ):
        return "reject", "Crop-level prompts rejected smoke inside the teacher region."

    if (
        negative_local
        and confident_local_negative
        and local_region_smoke == 1
        and local_region_confounder in atmospheric_confounders
    ):
        return "reject", "Crop-level prompts interpreted the teacher region as an atmospheric confounder."

    if (
        negative_full_frame
        and confident_negative
        and region_smoke == 0
    ):
        return "reject", "All full-frame prompts leaned away from smoke with confident negative evidence."

    if (
        negative_full_frame
        and confident_negative
        and region_smoke == 1
        and region_confounder in atmospheric_confounders
    ):
        return "reject", "Full-frame prompts rejected smoke and region reasoning only suggested an atmospheric confounder."

    # This project uses UAV wide-scene images, so a crop-only positive can be
    # misleading when the full frame is mixed or negative. The hybrid strategy
    # only fully accepts a crop when the wider scene also gives smoke support.
    if strong_local_positive and full_frame_strong_positive:
        return "accept", "Both crop-level and full-frame prompts strongly supported smoke."

    if strong_local_positive and full_frame_supportive:
        return "down_weight", "Crop-level prompts strongly supported smoke, but full-frame evidence was weaker."

    if strong_local_positive and negative_full_frame and confident_negative:
        return "down_weight", "Crop-level prompts supported smoke, but full-frame context argued against it."

    if strong_local_positive:
        return "uncertain", "Crop-level prompts supported smoke, but full-frame context was too weak to fully trust it."

    if strong_positive:
        return "accept", "All full-frame prompts strongly supported smoke."

    if (
        cautious_positive
        and mask_area >= down_weight_min_area
        and (
            uncertain_low <= (local_confidence_score or 0) < uncertain_high
            or local_region_stage == "early"
            or local_region_confounder in {"haze", "cloud", "fog"}
        )
    ):
        return "down_weight", "Crop-level prompts detected smoke, but local confidence/confounders suggest caution."

    if strong_positive and negative_local and confident_local_negative:
        return "uncertain", "Full-frame prompts supported smoke, but the local teacher region did not look reliable."

    # Mixed signals are kept separate so we can inspect them later.
    return "uncertain", "Signals were mixed, so the original teacher mask is kept."


def build_refinement_table(
    vlm_df: pd.DataFrame,
    sam_dir: Path | None = None,
    accept_threshold: int = 70,
    reject_threshold: int = 40,
    region_accept_threshold: int = 60,
    uncertain_low: int = 40,
    uncertain_high: int = 70,
    tiny_mask_threshold: int = 120,
    down_weight_min_area: int = 256,
    refinement_strategy: str = "hybrid",
) -> pd.DataFrame:
    full_frame = aggregate_region_rows(vlm_df, "full_frame", "full")
    teacher_crop = aggregate_region_rows(vlm_df, "teacher_box_context", "crop")
    if full_frame.empty and teacher_crop.empty:
        aggregated = pd.DataFrame(columns=["image_path", "image_stem"])
    elif full_frame.empty:
        aggregated = teacher_crop.copy()
    elif teacher_crop.empty:
        aggregated = full_frame.copy()
    else:
        aggregated = full_frame.merge(teacher_crop, on=["image_path", "image_stem"], how="outer")
    if sam_dir is not None:
        aggregated["mask_area"] = aggregated["image_stem"].apply(lambda stem: get_mask_area(stem, sam_dir))
    else:
        aggregated["mask_area"] = 0
    actions = aggregated.apply(
        decide_refinement_action,
        axis=1,
        result_type="expand",
        accept_threshold=accept_threshold,
        reject_threshold=reject_threshold,
        region_accept_threshold=region_accept_threshold,
        uncertain_low=uncertain_low,
        uncertain_high=uncertain_high,
        tiny_mask_threshold=tiny_mask_threshold,
        down_weight_min_area=down_weight_min_area,
        refinement_strategy=refinement_strategy,
    )
    aggregated["action"] = actions[0]
    aggregated["decision_reason"] = actions[1]
    aggregated["accept_threshold"] = accept_threshold
    aggregated["reject_threshold"] = reject_threshold
    aggregated["region_accept_threshold"] = region_accept_threshold
    aggregated["uncertain_low"] = uncertain_low
    aggregated["uncertain_high"] = uncertain_high
    aggregated["tiny_mask_threshold"] = tiny_mask_threshold
    aggregated["down_weight_min_area"] = down_weight_min_area
    aggregated["refinement_strategy"] = refinement_strategy
    return aggregated
