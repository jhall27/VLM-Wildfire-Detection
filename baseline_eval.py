"""
Comprehensive Baseline Evaluation
Combines segmentation (eval.py) and VLM (eval_with_vlm.py) results.
Generates tables, figures, and qualitative outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import shutil
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================================
# PART 1: LOAD RESULTS
# ============================================================================

def load_segmentation_results(seg_csv_path: Path) -> Dict[str, Any]:
    """Load segmentation metrics from eval.py output (CSV)."""
    df = pd.read_csv(seg_csv_path)

    # Convert CSV row to dictionary
    metrics = df.iloc[0].to_dict()
    return metrics


def load_vlm_results(vlm_csv_path: Path) -> pd.DataFrame:
    """Load parsed VLM results from eval_with_vlm.py output."""
    df = pd.read_csv(vlm_csv_path)
    return df


# ============================================================================
# PART 2: MERGE RESULTS
# ============================================================================

def merge_segmentation_and_vlm(
        seg_metrics: Dict[str, Any],
        vlm_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge segmentation metrics with VLM results."""

    merged_df = vlm_df.copy()

    # Add segmentation metrics (broadcast to all rows)
    seg_cols = {
        'seg_mean_iou': seg_metrics.get('mean_iou'),
        'seg_mean_f1': seg_metrics.get('mean_f1'),
        'seg_mean_precision': seg_metrics.get('mean_precision'),
        'seg_mean_recall': seg_metrics.get('mean_recall'),
        'seg_mean_rand': seg_metrics.get('mean_rand'),
        'seg_mean_loss': seg_metrics.get('mean_val_loss'),
        'seg_fps': seg_metrics.get('fps'),
        'seg_ms_per_image': seg_metrics.get('milliseconds_per_image'),
        'seg_model': seg_metrics.get('model'),
        'seg_dataset': seg_metrics.get('dataset'),
    }

    for col_name, value in seg_cols.items():
        merged_df[col_name] = value

    return merged_df


# ============================================================================
# PART 3: GENERATE BASELINE TABLES
# ============================================================================

def generate_segmentation_table(seg_metrics: Dict[str, Any], output_dir: Path) -> pd.DataFrame:
    """Create segmentation metrics summary table."""

    table_data = {
        'Metric': [
            'Mean IoU',
            'Mean F1 Score',
            'Mean Precision',
            'Mean Recall',
            'Mean Rand Index',
            'Mean Validation Loss',
            'FPS',
            'Milliseconds per Image',
        ],
        'Value': [
            f"{seg_metrics.get('mean_iou', 0):.4f}",
            f"{seg_metrics.get('mean_f1', 0):.4f}",
            f"{seg_metrics.get('mean_precision', 0):.4f}",
            f"{seg_metrics.get('mean_recall', 0):.4f}",
            f"{seg_metrics.get('mean_rand', 0):.4f}",
            f"{seg_metrics.get('mean_val_loss', 0):.6f}",
            f"{seg_metrics.get('fps', 0):.2f}",
            f"{seg_metrics.get('milliseconds_per_image', 0):.2f}",
        ]
    }

    seg_table = pd.DataFrame(table_data)
    seg_table.to_csv(output_dir / 'segmentation_metrics_table.csv', index=False)

    return seg_table


def generate_vlm_table(vlm_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Create VLM evaluation summary table."""

    summary_data = {
        'Metric': [
            'Total Evaluations',
            'Unique Images',
            'Smoke Detected (Count)',
            'Smoke Detection Rate (%)',
            'Avg Confidence Score',
            'Confidence Score Std Dev',
        ],
        'Value': [
            len(vlm_df),
            vlm_df['image_path'].nunique() if 'image_path' in vlm_df.columns else 'N/A',
            int(vlm_df['smoke_detected'].sum()) if 'smoke_detected' in vlm_df.columns else 0,
            f"{(vlm_df['smoke_detected'].mean() * 100):.1f}" if 'smoke_detected' in vlm_df.columns else 'N/A',
            f"{vlm_df['confidence'].mean():.1f}" if 'confidence' in vlm_df.columns else 'N/A',
            f"{vlm_df['confidence'].std():.1f}" if 'confidence' in vlm_df.columns else 'N/A',
        ]
    }

    vlm_table = pd.DataFrame(summary_data)
    vlm_table.to_csv(output_dir / 'vlm_metrics_table.csv', index=False)

    return vlm_table


def generate_combined_summary_table(
        seg_metrics: Dict[str, Any],
        vlm_df: pd.DataFrame,
        output_dir: Path
) -> pd.DataFrame:
    """Create combined summary table."""

    summary = {
        'Component': ['Segmentation', 'VLM'],
        'Primary Metric': [
            f"IoU: {seg_metrics.get('mean_iou', 0):.4f}",
            f"Detection Rate: {(vlm_df['smoke_detected'].mean() * 100):.1f}%" if 'smoke_detected' in vlm_df.columns else 'N/A'
        ],
        'Secondary Metric': [
            f"F1: {seg_metrics.get('mean_f1', 0):.4f}",
            f"Avg Confidence: {vlm_df['confidence'].mean():.1f}%" if 'confidence' in vlm_df.columns else 'N/A'
        ],
        'Runtime': [
            f"{seg_metrics.get('fps', 0):.2f} FPS",
            f"{len(vlm_df)} evaluations"
        ]
    }

    summary_table = pd.DataFrame(summary)
    summary_table.to_csv(output_dir / 'baseline_summary_table.csv', index=False)

    return summary_table


# ============================================================================
# PART 4: GENERATE BASELINE FIGURES
# ============================================================================

def generate_segmentation_figures(seg_metrics: Dict[str, Any], output_dir: Path) -> None:
    """Generate visualization of segmentation metrics."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Segmentation Metrics - {seg_metrics.get('model', 'Unknown')}", fontsize=16)

    # Quality metrics
    metrics = ['IoU', 'F1', 'Precision', 'Recall']
    values = [
        seg_metrics.get('mean_iou', 0),
        seg_metrics.get('mean_f1', 0),
        seg_metrics.get('mean_precision', 0),
        seg_metrics.get('mean_recall', 0),
    ]

    axes[0, 0].bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[0, 0].set_title('Quality Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)

    # Loss
    axes[0, 1].bar(['Val Loss'], [seg_metrics.get('mean_val_loss', 0)], color='#9467bd')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_ylabel('Loss Value')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Runtime metrics
    runtime_metrics = ['FPS', 'MS/Image']
    runtime_values = [
        seg_metrics.get('fps', 0),
        seg_metrics.get('milliseconds_per_image', 0) / 1000
    ]
    axes[1, 0].bar(runtime_metrics, runtime_values, color=['#17becf', '#bcbd22'])
    axes[1, 0].set_title('Runtime Performance')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(axis='y', alpha=0.3)

    # Rand Index
    axes[1, 1].bar(['Rand Index'], [seg_metrics.get('mean_rand', 0)], color='#e377c2')
    axes[1, 1].set_title('Rand Index')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'segmentation_metrics_figure.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved segmentation figure: {output_dir / 'segmentation_metrics_figure.png'}")
    plt.close()


def generate_vlm_figures(vlm_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate visualization of VLM results."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('VLM Evaluation Results', fontsize=16)

    # Smoke detection distribution
    if 'smoke_detected' in vlm_df.columns:
        smoke_counts = vlm_df['smoke_detected'].value_counts()
        smoke_labels = ['Smoke Not Detected', 'Smoke Detected']
        smoke_values = [
            smoke_counts.get(0, 0),
            smoke_counts.get(1, 0)
        ]
        colors = ['#2ca02c', '#d62728']
        axes[0].pie(smoke_values, labels=smoke_labels, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0].set_title('Smoke Detection Distribution')

    # Confidence distribution
    if 'confidence' in vlm_df.columns:
        confidence_vals = vlm_df['confidence'].dropna()
        axes[1].hist(confidence_vals, bins=20, color='#1f77b4', edgecolor='black', alpha=0.7)
        axes[1].set_title('Confidence Score Distribution')
        axes[1].set_xlabel('Confidence Score (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'vlm_results_figure.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved VLM figure: {output_dir / 'vlm_results_figure.png'}")
    plt.close()


# ============================================================================
# PART 5: SAVE QUALITATIVE OUTPUTS
# ============================================================================

def save_failure_cases_summary(merged_df: pd.DataFrame, output_dir: Path) -> None:
    """Identify and save failure cases with mask references."""

    masks_dir = Path('seg_outputs/masks').resolve()

    if 'confidence' in merged_df.columns and 'smoke_detected' in merged_df.columns:
        # Low confidence cases
        low_conf = merged_df[merged_df['confidence'] < 50].copy()
        if len(low_conf) > 0:
            # Add mask path reference using image filename
            if 'image_path' in low_conf.columns:
                low_conf['mask_path'] = low_conf['image_path'].apply(
                    lambda x: str(masks_dir / f"{Path(x).stem}_mask.png")
                )
            low_conf.to_csv(output_dir / 'failure_low_confidence_cases.csv', index=False)
            print(f"✓ Saved low confidence cases: {len(low_conf)} cases")

        # Borderline confidence cases
        mid_conf = merged_df[(merged_df['confidence'] >= 40) & (merged_df['confidence'] <= 60)].copy()
        if len(mid_conf) > 0:
            # Add mask path reference using image filename
            if 'image_path' in mid_conf.columns:
                mid_conf['mask_path'] = mid_conf['image_path'].apply(
                    lambda x: str(masks_dir / f"{Path(x).stem}_mask.png")
                )
            mid_conf.to_csv(output_dir / 'failure_borderline_confidence_cases.csv', index=False)
            print(f"✓ Saved borderline cases: {len(mid_conf)} cases")


def save_success_cases_summary(merged_df: pd.DataFrame, output_dir: Path) -> None:
    """Save success cases - high confidence predictions with mask references."""

    masks_dir = Path('seg_outputs/masks').resolve()

    if 'confidence' in merged_df.columns:
        success_cases = merged_df[merged_df['confidence'] >= 75].copy()
        if len(success_cases) > 0:
            # Add mask path reference using image filename
            if 'image_path' in success_cases.columns:
                success_cases['mask_path'] = success_cases['image_path'].apply(
                    lambda x: str(masks_dir / f"{Path(x).stem}_mask.png")
                )
            success_cases.to_csv(output_dir / 'success_high_confidence_cases.csv', index=False)
            print(f"✓ Saved success cases: {len(success_cases)} high-confidence predictions")


def save_comprehensive_analysis(merged_df: pd.DataFrame, output_dir: Path) -> None:
    """Save full merged results."""

    merged_df.to_csv(output_dir / 'comprehensive_baseline_results.csv', index=False)
    print(f"✓ Saved comprehensive results: {len(merged_df)} total evaluations")


# ============================================================================
# PART 5B: GENERATE MASK IMAGES VIA MODEL INFERENCE
# ============================================================================

def _inverse_norm(img_tensor):
    """Undo ImageNet normalisation so pixel values are back in [0, 1]."""
    from torchvision.transforms import v2
    img = v2.functional.normalize(
        img_tensor,
        mean=[0.0, 0.0, 0.0],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    img = v2.functional.normalize(
        img,
        mean=[-0.485, -0.456, -0.406],
        std=[1.0, 1.0, 1.0],
    )
    return img


def generate_masks_for_test_images(
    model_exp: str = "sam_sup_pidnet_s",
    model_size: str = "s",
    weight_dir: str = "checkpoints",
    data_dir: str = "data",
    output_dir: Path = Path("baseline_eval_output"),
    mode: str = "test",
    device: str = "cpu",
    max_images: Optional[int] = None,
) -> Dict[str, Dict]:
    """
    Run PIDNet inference on the test split and save mask artefacts.

    For every image the function saves:
      - A binary mask PNG  → output_dir/masks/<id>_mask.png
      - A 3-panel overlay  → output_dir/overlays/<id>_overlay.png
        (original | ground-truth | prediction with IoU in title)

    Returns
    -------
    dict  {image_id: {"iou": float, "mask_path": str, "overlay_path": str}}
    """
    import torch
    import torch.nn.functional as F
    import einops
    from PIL import Image as PILImage
    from torch.utils.data import DataLoader
    from torchmetrics.classification import BinaryJaccardIndex

    # Make project imports available when running from outside the sub-directory.
    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from pidnet_utils.configs import config as pidnet_config
    from models.pidnet import get_seg_model
    from pidnet_utils.criterion import BondaryLoss
    from pidnet_utils.utils import FullModel
    from datasets.segmentation_data import WFSeg

    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build minimal args namespace expected by WFSeg ----
    wf_args = SimpleNamespace(
        eval_sam=False,
        eval_snake=False,
        sup_dir="sam_masks",
        eval_dir="manual_masks",
        include_id=True,
        include_box=False,
        label_format="txt",
        boxsup=False,
    )

    # ---- Load model ----
    pidnet_config.MODEL.NAME = "pidnet_" + model_size
    pidnet_config.MODEL.PRETRAINED = (
        f"pretrained_models/imagenet/PIDNet_{model_size.capitalize()}_ImageNet.pth.tar"
    )
    base_model = get_seg_model(cfg=pidnet_config, imgnet_pretrained=True)
    pos_weight = einops.rearrange(torch.tensor([1]), "(a b c) -> a b c", a=1, b=1)
    sem_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bd_criterion = BondaryLoss()
    model = FullModel(base_model, sem_criterion, bd_criterion)

    weight_path = os.path.join(weight_dir, model_exp + ".pt")
    weights = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(weights, strict=False)
    model.eval()
    model.to(device)
    print(f"      Loaded weights: {weight_path}")

    # ---- Dataset ----
    dataset = WFSeg(data_dir, mode=mode, manual_masks=True, boundary=True, args=wf_args)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"      Dataset size: {len(dataset)} images (mode={mode})")

    iou_fn = BinaryJaccardIndex(zero_division=1.0e-9)
    results: Dict[str, Dict] = {}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_images is not None and i >= max_images:
                break

            img_tensor = batch["img"].to(device)
            gt_mask = batch["mask"].to(device)

            # Image id is stored with a potential trailing char from [:-4] on .jpeg
            raw_id = batch["id"][0] if "id" in batch else str(i)
            img_id = Path(raw_id).stem if "." in raw_id else raw_id

            # Forward pass (plot_outputs=True skips the loss path)
            outputs = model(inputs=img_tensor, plot_outputs=True)
            pred_sigmoid = F.sigmoid(outputs[1])
            pred_mask = torch.round(pred_sigmoid)

            # Per-image IoU against ground-truth mask
            iou_val = iou_fn(pred_mask.cpu(), gt_mask.cpu()).item()

            # ---- Save binary mask PNG ----
            mask_np = pred_mask[0, 0].cpu().numpy().astype(np.uint8) * 255
            mask_path = masks_dir / f"{img_id}_mask.png"
            PILImage.fromarray(mask_np).save(mask_path)

            # ---- Save 3-panel overlay ----
            rgb = einops.rearrange(
                _inverse_norm(img_tensor[0]).cpu().numpy(), "c h w -> h w c"
            )
            rgb = np.clip(rgb, 0.0, 1.0)
            gt_np = gt_mask[0, 0].cpu().numpy()
            pred_np = pred_mask[0, 0].cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].imshow(rgb)
            axes[0].set_title("Original", fontsize=11)
            axes[0].axis("off")

            axes[1].imshow(rgb)
            axes[1].imshow(gt_np, alpha=0.45, cmap="Reds", vmin=0, vmax=1)
            axes[1].set_title("Ground Truth", fontsize=11)
            axes[1].axis("off")

            axes[2].imshow(rgb)
            axes[2].imshow(pred_np, alpha=0.45, cmap="Blues", vmin=0, vmax=1)
            axes[2].set_title(f"Prediction  (IoU = {iou_val:.3f})", fontsize=11)
            axes[2].axis("off")

            fig.suptitle(img_id, fontsize=10, y=1.01)
            plt.tight_layout()
            overlay_path = overlays_dir / f"{img_id}_overlay.png"
            plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            results[img_id] = {
                "iou": iou_val,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }

            if (i + 1) % 10 == 0:
                print(f"      ... {i + 1} images processed")

    print(f"      ✓ Generated {len(results)} masks and overlays")
    return results


# ============================================================================
# PART 5C: QUALITATIVE MASK PANELS (success / failure grids)
# ============================================================================

def save_qualitative_mask_panels(
    per_image_results: Dict[str, Dict],
    output_dir: Path,
    n_success: int = 5,
    n_failure: int = 5,
    iou_success_thresh: float = 0.40,
    iou_failure_thresh: float = 0.10,
) -> None:
    """
    Create and save multi-panel grids for the best (success) and worst
    (failure) predictions, sorted by IoU.

    Also copies individual overlays into the matching sub-folder and writes
    a ``per_image_iou.csv`` ranking every image.

    Parameters
    ----------
    per_image_results : dict returned by ``generate_masks_for_test_images``
    output_dir        : root output directory
    n_success / n_failure : how many images to include in each panel
    iou_success_thresh    : minimum IoU to be called a success
    iou_failure_thresh    : maximum IoU to be called a failure
    """
    output_dir = Path(output_dir)
    success_dir = output_dir / "qualitative" / "success"
    failure_dir = output_dir / "qualitative" / "failure"
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    sorted_items: List[Tuple[str, Dict]] = sorted(
        per_image_results.items(), key=lambda x: x[1]["iou"]
    )

    failure_cases = [
        (img_id, data)
        for img_id, data in sorted_items
        if data["iou"] <= iou_failure_thresh
    ][:n_failure]

    success_cases = [
        (img_id, data)
        for img_id, data in reversed(sorted_items)
        if data["iou"] >= iou_success_thresh
    ][:n_success]

    def _make_panel(cases: List[Tuple[str, Dict]], title: str, save_path: Path) -> None:
        if not cases:
            print(f"      ! No cases for panel: {title}")
            return
        n = len(cases)
        fig, axes = plt.subplots(n, 1, figsize=(18, 5 * n))
        if n == 1:
            axes = [axes]
        for ax, (img_id, data) in zip(axes, cases):
            overlay_img = plt.imread(data["overlay_path"])
            ax.imshow(overlay_img)
            ax.set_title(f"{img_id}   |   IoU = {data['iou']:.3f}", fontsize=11)
            ax.axis("off")
        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"      ✓ Saved panel: {save_path}")

    _make_panel(
        success_cases,
        f"Success Cases — Top {len(success_cases)} by IoU (≥ {iou_success_thresh:.2f})",
        success_dir / "success_panel.png",
    )
    _make_panel(
        failure_cases,
        f"Failure Cases — Bottom {len(failure_cases)} by IoU (≤ {iou_failure_thresh:.2f})",
        failure_dir / "failure_panel.png",
    )

    # Copy individual overlays into their sub-folder for easy inspection
    for img_id, data in success_cases:
        dest = success_dir / Path(data["overlay_path"]).name
        shutil.copy(data["overlay_path"], dest)
    for img_id, data in failure_cases:
        dest = failure_dir / Path(data["overlay_path"]).name
        shutil.copy(data["overlay_path"], dest)

    # ---- Binary mask contact sheet ----
    _save_mask_contact_sheet(per_image_results, output_dir)

    # ---- Per-image IoU CSV ----
    rows = [{"image_id": k, **v} for k, v in per_image_results.items()]
    iou_csv = output_dir / "per_image_iou.csv"
    pd.DataFrame(rows).sort_values("iou").to_csv(iou_csv, index=False)
    print(f"      ✓ Saved per-image IoU rankings: {iou_csv}")


def _save_mask_contact_sheet(
    per_image_results: Dict[str, Dict],
    output_dir: Path,
    n_cols: int = 5,
    max_masks: int = 25,
) -> None:
    """
    Tile binary mask thumbnails into a single contact-sheet image so the
    spread of predictions is visible at a glance.
    """
    from PIL import Image as PILImage

    items = list(per_image_results.items())[:max_masks]
    if not items:
        return

    thumbs = []
    labels = []
    thumb_size = (192, 108)  # 1/10th of 1920×1080

    for img_id, data in items:
        mask_path = Path(data["mask_path"])
        if not mask_path.exists():
            continue
        mask = PILImage.open(mask_path).convert("L").resize(thumb_size, PILImage.NEAREST)
        thumbs.append(np.array(mask))
        labels.append(f"{img_id[:18]}\nIoU={data['iou']:.2f}")

    if not thumbs:
        return

    n_rows = int(np.ceil(len(thumbs) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.0)
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, (thumb, label) in enumerate(zip(thumbs, labels)):
        r, c = divmod(idx, n_cols)
        axes[r, c].imshow(thumb, cmap="gray", vmin=0, vmax=255)
        axes[r, c].set_title(label, fontsize=6)
        axes[r, c].axis("off")

    # Hide unused axes
    for idx in range(len(thumbs), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("Predicted Binary Masks — Contact Sheet", fontsize=12, fontweight="bold")
    plt.tight_layout()
    sheet_path = output_dir / "masks_contact_sheet.png"
    plt.savefig(sheet_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      ✓ Saved mask contact sheet: {sheet_path}")


# ============================================================================
# PART 5D: TOP-LEVEL MASK QUALITATIVE PIPELINE
# ============================================================================

def compile_mask_qualitative_outputs(
    model_exp: str = "sam_sup_pidnet_s",
    model_size: str = "s",
    weight_dir: str = "checkpoints",
    data_dir: str = "data",
    output_dir: Path = Path("baseline_eval_output"),
    mode: str = "test",
    device: str = "cpu",
    max_images: Optional[int] = None,
    n_success: int = 5,
    n_failure: int = 5,
) -> Dict[str, Dict]:
    """
    End-to-end mask qualitative pipeline:
      1. Run inference → save binary masks + 3-panel overlays
      2. Build success/failure panels and a contact sheet

    Can be called standalone (no VLM results needed).

    Returns the per-image results dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("MASK QUALITATIVE OUTPUT PIPELINE")
    print("=" * 70)

    print(f"\n[1/2] Generating masks and overlays for '{mode}' split ...")
    per_image = generate_masks_for_test_images(
        model_exp=model_exp,
        model_size=model_size,
        weight_dir=weight_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        mode=mode,
        device=device,
        max_images=max_images,
    )

    print(f"\n[2/2] Building qualitative panels ...")
    save_qualitative_mask_panels(
        per_image_results=per_image,
        output_dir=output_dir,
        n_success=n_success,
        n_failure=n_failure,
    )

    mean_iou = float(np.mean([v["iou"] for v in per_image.values()])) if per_image else 0.0
    print("\n" + "=" * 70)
    print("MASK QUALITATIVE PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Images processed : {len(per_image)}")
    print(f"  Mean IoU         : {mean_iou:.4f}")
    print(f"  Output directory : {output_dir.absolute()}")
    print(f"    masks/          — binary mask PNGs")
    print(f"    overlays/       — 3-panel overlay PNGs")
    print(f"    qualitative/    — success & failure panels")
    print(f"    masks_contact_sheet.png")
    print(f"    per_image_iou.csv")
    print("=" * 70 + "\n")

    return per_image


# ============================================================================
# PART 6: MAIN COMPILATION FUNCTION
# ============================================================================

def compile_baseline_evaluation(
        seg_csv_path: Path,
        vlm_csv_path: Path,
        output_dir: Path = Path('baseline_eval_output'),
        # Optional mask qualitative args — set model_exp to trigger mask generation
        model_exp: Optional[str] = None,
        model_size: str = "s",
        weight_dir: str = "checkpoints",
        data_dir: str = "data",
        mode: str = "test",
        device: str = "cpu",
        max_images: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compile complete baseline evaluation.

    Args:
        seg_csv_path : Path to segmentation metrics CSV from eval.py
        vlm_csv_path : Path to VLM results CSV from eval_with_vlm.py
        output_dir   : Output directory for all results
        model_exp    : If set, run mask inference and save qualitative visuals.
                       Should match the checkpoint stem, e.g. "sam_sup_pidnet_s".
        model_size   : PIDNet size ("s", "m", or "l").
        weight_dir   : Directory containing .pt checkpoint files.
        data_dir     : Root of the wildfire dataset.
        mode         : Dataset split to visualise ("test" or "valid").
        device       : Torch device string ("cpu" or "cuda").
        max_images   : Cap on images for mask generation (None = all).

    Returns:
        Tuple of (merged_dataframe, summary_dict)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("BASELINE EVALUATION COMPILATION")
    print("=" * 70)

    # Load results
    print("\n[1/7] Loading segmentation results...")
    seg_metrics = load_segmentation_results(seg_csv_path)
    print(f"      ✓ Model: {seg_metrics.get('model')}")
    print(f"      ✓ Mean IoU: {seg_metrics.get('mean_iou'):.4f}")

    print("\n[2/7] Loading VLM results...")
    vlm_df = load_vlm_results(vlm_csv_path)
    print(f"      ✓ Evaluations loaded: {len(vlm_df)}")

    # Merge
    print("\n[3/7] Merging segmentation and VLM results...")
    merged_df = merge_segmentation_and_vlm(seg_metrics, vlm_df)
    print(f"      ✓ Merged dataframe shape: {merged_df.shape}")

    # Generate tables
    print("\n[4/7] Generating baseline tables...")
    seg_table = generate_segmentation_table(seg_metrics, output_dir)
    vlm_table = generate_vlm_table(vlm_df, output_dir)
    summary_table = generate_combined_summary_table(seg_metrics, vlm_df, output_dir)
    print(f"      ✓ Saved 3 table files")

    # Generate figures
    print("\n[5/7] Generating baseline figures...")
    generate_segmentation_figures(seg_metrics, output_dir)
    generate_vlm_figures(vlm_df, output_dir)
    print(f"      ✓ Saved 2 figure files")

    # Qualitative outputs (VLM-confidence-based CSVs)
    print("\n[6/7] Saving qualitative analysis...")
    save_failure_cases_summary(merged_df, output_dir)
    save_success_cases_summary(merged_df, output_dir)
    save_comprehensive_analysis(merged_df, output_dir)
    print(f"      ✓ Saved qualitative outputs")

    # Optional: mask-level qualitative visuals
    per_image: Dict[str, Dict] = {}
    if model_exp is not None:
        print("\n[6b] Generating mask qualitative outputs...")
        per_image = generate_masks_for_test_images(
            model_exp=model_exp,
            model_size=model_size,
            weight_dir=weight_dir,
            data_dir=data_dir,
            output_dir=output_dir,
            mode=mode,
            device=device,
            max_images=max_images,
        )
        save_qualitative_mask_panels(per_image, output_dir)

    # Summary JSON
    print("\n[7/7] Creating summary report...")
    mean_iou_per_img = (
        float(np.mean([v["iou"] for v in per_image.values()])) if per_image else None
    )
    summary = {
        'timestamp': datetime.now().isoformat(),
        'segmentation': {
            'model': seg_metrics.get('model'),
            'dataset': seg_metrics.get('dataset'),
            'metrics': {k: v for k, v in seg_metrics.items() if k not in ['model', 'dataset']}
        },
        'vlm': {
            'total_evaluations': len(vlm_df),
            'unique_images': vlm_df['image_path'].nunique() if 'image_path' in vlm_df.columns else None,
            'smoke_detection_rate': float(
                vlm_df['smoke_detected'].mean()) if 'smoke_detected' in vlm_df.columns else None,
            'avg_confidence': float(vlm_df['confidence'].mean()) if 'confidence' in vlm_df.columns else None,
        },
        'mask_qualitative': {
            'images_visualised': len(per_image),
            'mean_iou_per_image': mean_iou_per_img,
        },
        'output_files': {
            'tables': ['segmentation_metrics_table.csv', 'vlm_metrics_table.csv', 'baseline_summary_table.csv'],
            'figures': ['segmentation_metrics_figure.png', 'vlm_results_figure.png'],
            'data': ['comprehensive_baseline_results.csv', 'failure_*.csv', 'success_*.csv'],
            'masks': ['masks/<id>_mask.png', 'overlays/<id>_overlay.png',
                      'qualitative/success/success_panel.png',
                      'qualitative/failure/failure_panel.png',
                      'masks_contact_sheet.png', 'per_image_iou.csv'],
        }
    }

    with open(output_dir / 'baseline_evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final report
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nSegmentation Model: {seg_metrics.get('model')}")
    print(f"  Mean IoU:     {seg_metrics.get('mean_iou'):.4f}")
    print(f"  Mean F1:      {seg_metrics.get('mean_f1'):.4f}")
    print(f"  FPS:          {seg_metrics.get('fps'):.2f}")
    print(f"\nVLM Evaluation:")
    print(f"  Evaluations:  {len(vlm_df)}")
    print(f"  Unique Images: {vlm_df['image_path'].nunique() if 'image_path' in vlm_df.columns else 'N/A'}")
    print(
        f"  Smoke Rate:   {(vlm_df['smoke_detected'].mean() * 100):.1f}%" if 'smoke_detected' in vlm_df.columns else "  Smoke Rate:   N/A")
    print(
        f"  Avg Conf:     {vlm_df['confidence'].mean():.1f}%" if 'confidence' in vlm_df.columns else "  Avg Conf:     N/A")
    if per_image:
        print(f"\nMask Qualitative Outputs:")
        print(f"  Images visualised: {len(per_image)}")
        print(f"  Mean per-image IoU: {mean_iou_per_img:.4f}")
    print(f"\nOutput Directory: {output_dir.absolute()}")
    print("=" * 70 + "\n")

    return merged_df, summary


def main() -> None:
    """
    Main execution.

    Supports two modes controlled by the MASKS_ONLY flag below:

    Mode A (default): compile tables, figures, and VLM-based qualitative CSVs.
      Requires eval_results.csv and vlm_outputs/vlm_results_parsed.csv.

    Mode B (MASKS_ONLY=True): generate binary masks, 3-panel overlays, and
      success/failure panels directly from the model — no VLM CSV needed.
    """
    # ------------------------------------------------------------------ #
    # Configuration — edit these lines to match your run.                 #
    # ------------------------------------------------------------------ #
    MASKS_ONLY   = False           # Set True to skip VLM step
    MODEL_EXP    = "sam_sup_pidnet_s"   # Checkpoint stem in checkpoints/
    MODEL_SIZE   = "s"             # "s", "m", or "l"
    WEIGHT_DIR   = "checkpoints"
    DATA_DIR     = "data"
    MODE         = "test"          # Dataset split
    DEVICE       = "cpu"           # "cpu" or "cuda"
    MAX_IMAGES   = None            # Set an int to limit (e.g. 20 for a quick test)

    output_dir   = Path("baseline_eval_output")
    seg_csv_path = Path("eval_results.csv")
    vlm_csv_path = Path("vlm_outputs/vlm_results_parsed.csv")
    # ------------------------------------------------------------------ #

    if MASKS_ONLY:
        compile_mask_qualitative_outputs(
            model_exp=MODEL_EXP,
            model_size=MODEL_SIZE,
            weight_dir=WEIGHT_DIR,
            data_dir=DATA_DIR,
            output_dir=output_dir,
            mode=MODE,
            device=DEVICE,
            max_images=MAX_IMAGES,
        )
        return

    # Full pipeline
    if not seg_csv_path.exists():
        print(f"\n✗ Error: Segmentation results not found: {seg_csv_path}")
        print(f"  Make sure you've run: python eval.py")
        return

    if not vlm_csv_path.exists():
        print(f"\n✗ Error: VLM results not found: {vlm_csv_path}")
        print(f"  Make sure you've run: python eval_with_vlm.py")
        return

    compile_baseline_evaluation(
        seg_csv_path=seg_csv_path,
        vlm_csv_path=vlm_csv_path,
        output_dir=output_dir,
        model_exp=MODEL_EXP,   # pass None to skip mask generation
        model_size=MODEL_SIZE,
        weight_dir=WEIGHT_DIR,
        data_dir=DATA_DIR,
        mode=MODE,
        device=DEVICE,
        max_images=MAX_IMAGES,
    )


if __name__ == "__main__":
    main()
