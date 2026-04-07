"""
Comprehensive Baseline Evaluation
Combines segmentation (eval.py) and VLM (eval_with_vlm.py) results.
Generates tables, figures, and qualitative outputs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, Any
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
    """Identify and save failure cases."""

    if 'confidence' in merged_df.columns and 'smoke_detected' in merged_df.columns:
        # Low confidence cases
        low_conf = merged_df[merged_df['confidence'] < 50]
        if len(low_conf) > 0:
            low_conf.to_csv(output_dir / 'failure_low_confidence_cases.csv', index=False)
            print(f"✓ Saved low confidence cases: {len(low_conf)} cases")

        # Borderline confidence cases
        mid_conf = merged_df[(merged_df['confidence'] >= 40) & (merged_df['confidence'] <= 60)]
        if len(mid_conf) > 0:
            mid_conf.to_csv(output_dir / 'failure_borderline_confidence_cases.csv', index=False)
            print(f"✓ Saved borderline cases: {len(mid_conf)} cases")


def save_success_cases_summary(merged_df: pd.DataFrame, output_dir: Path) -> None:
    """Save success cases - high confidence predictions."""

    if 'confidence' in merged_df.columns:
        success_cases = merged_df[merged_df['confidence'] >= 75]
        if len(success_cases) > 0:
            success_cases.to_csv(output_dir / 'success_high_confidence_cases.csv', index=False)
            print(f"✓ Saved success cases: {len(success_cases)} high-confidence predictions")


def save_comprehensive_analysis(merged_df: pd.DataFrame, output_dir: Path) -> None:
    """Save full merged results."""

    merged_df.to_csv(output_dir / 'comprehensive_baseline_results.csv', index=False)
    print(f"✓ Saved comprehensive results: {len(merged_df)} total evaluations")


# ============================================================================
# PART 6: MAIN COMPILATION FUNCTION
# ============================================================================

def compile_baseline_evaluation(
        seg_csv_path: Path,
        vlm_csv_path: Path,
        output_dir: Path = Path('baseline_eval_output'),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Compile complete baseline evaluation.

    Args:
        seg_csv_path: Path to segmentation metrics CSV from eval.py
        vlm_csv_path: Path to VLM results CSV from eval_with_vlm.py
        output_dir: Output directory for all results

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

    # Qualitative outputs
    print("\n[6/7] Saving qualitative analysis...")
    save_failure_cases_summary(merged_df, output_dir)
    save_success_cases_summary(merged_df, output_dir)
    save_comprehensive_analysis(merged_df, output_dir)
    print(f"      ✓ Saved qualitative outputs")

    # Summary JSON
    print("\n[7/7] Creating summary report...")
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
        'output_files': {
            'tables': ['segmentation_metrics_table.csv', 'vlm_metrics_table.csv', 'baseline_summary_table.csv'],
            'figures': ['segmentation_metrics_figure.png', 'vlm_results_figure.png'],
            'data': ['comprehensive_baseline_results.csv', 'failure_*.csv', 'success_*.csv'],
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
    print(f"\nOutput Directory: {output_dir.absolute()}")
    print("=" * 70 + "\n")

    return merged_df, summary


def main() -> None:
    """Main execution - Compile baseline evaluation."""
    seg_csv_path = Path('eval_results.csv')
    vlm_csv_path = Path('vlm_outputs/vlm_results_parsed.csv')
    output_dir = Path('baseline_eval_output')

    # Check if files exist
    if not seg_csv_path.exists():
        print(f"\n✗ Error: Segmentation results not found: {seg_csv_path}")
        print(f"  Make sure you've run: python eval.py")
        return

    if not vlm_csv_path.exists():
        print(f"\n✗ Error: VLM results not found: {vlm_csv_path}")
        print(f"  Make sure you've run: python eval_with_vlm.py")
        return

    # Compile baseline evaluation
    merged_df, summary = compile_baseline_evaluation(
        seg_csv_path=seg_csv_path,
        vlm_csv_path=vlm_csv_path,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()