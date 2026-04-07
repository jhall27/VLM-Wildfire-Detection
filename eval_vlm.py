"""Integrate VLM results into baseline evaluation."""
import argparse

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import defaultdict
import re


def parse_vlm_response(response: str, prompt_style: str) -> dict:
    """Parse VLM response based on prompt style."""
    result = {
        "raw_response": response,
        "smoke_detected": None,
        "smoke_stage": None,
        "confidence": None,
        "label": None,
        "confounder": None,
    }

    # Handle empty responses
    if pd.isna(response) or response == "":
        return result

    if prompt_style == "yes_no":
        # Simple yes/no extraction
        response_lower = response.lower().strip()
        if "yes" in response_lower:
            result["smoke_detected"] = 1
        elif "no" in response_lower:
            result["smoke_detected"] = 0

    elif prompt_style == "confidence_score":
        # Parse structured format:
        # label: <label>
        # confidence: <0-100>
        # reason: <short reason>
        lines = response.split('\n')
        for line in lines:
            if 'label:' in line.lower():
                #Only checks if smoke is in label (other labels are not considered for a fire)
                label = line.split(':', 1)[1].strip().lower()
                result["label"] = label
                result["smoke_detected"] = 1 if "smoke" in label else 0
            elif 'confidence:' in line.lower():
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    conf = int(re.search(r'\d+', conf_str).group())
                    result["confidence"] = min(max(conf, 0), 100)
                except:
                    pass

    elif prompt_style == "region_reasoning":
        # Parse structured format:
        # smoke_present: <yes/no>
        # smoke_stage: <early/obvious/none>
        # confounder: <cloud/fog/haze/background/none>
        # confidence: <0-100>
        # reason: <short reason>
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'smoke_present:' in line_lower:
                val = line.split(':', 1)[1].strip().lower()
                result["smoke_detected"] = 1 if "yes" in val else (0 if "no" in val else None)
            elif 'smoke_stage:' in line_lower:
                stage = line.split(':', 1)[1].strip().lower()
                result["smoke_stage"] = stage
            elif 'confounder:' in line_lower:
                confounder = line.split(':', 1)[1].strip().lower()
                result["confounder"] = confounder
            elif 'confidence:' in line_lower:
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    conf = int(re.search(r'\d+', conf_str).group())
                    result["confidence"] = min(max(conf, 0), 100)
                except:
                    pass

    return result


def load_vlm_results(vlm_csv_path: Path) -> pd.DataFrame:
    """Load VLM pilot results and parse responses."""
    df = pd.read_csv(vlm_csv_path)

    parsed_rows = []
    for _, row in df.iterrows():
        parsed = parse_vlm_response(row['response'], row['prompt_style'])
        row_dict = row.to_dict()
        row_dict.update(parsed)
        parsed_rows.append(row_dict)

    return pd.DataFrame(parsed_rows)


def generate_vlm_evaluation_report(vlm_df: pd.DataFrame, output_dir: Path = None) -> dict:
    """Generate evaluation report for VLM performance."""
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_images": len(vlm_df),
        "unique_images": vlm_df['image_path'].nunique() if 'image_path' in vlm_df.columns else None,
        "smoke_detection_rate": float(vlm_df['smoke_detected'].mean()) if 'smoke_detected' in vlm_df.columns else None,
        "avg_confidence": float(vlm_df['confidence'].mean()) if 'confidence' in vlm_df.columns else None,
        "model_name": vlm_df['model_name'].iloc[0] if 'model_name' in vlm_df.columns and len(vlm_df) > 0 else None,
    }

    # Save summary
    if output_dir:
        with open(output_dir / "vlm_evaluation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\n=== VLM Evaluation Summary ===")
    print(f"Total images evaluated: {summary['total_images']}")
    if summary['unique_images']:
        print(f"Unique images: {summary['unique_images']}")
    if summary['model_name']:
        print(f"Model: {summary['model_name']}")
    print(f"Smoke detection rate: {summary['smoke_detection_rate']:.1%}" if summary[
                                                                                'smoke_detection_rate'] is not None else "Smoke detection rate: N/A")
    print(f"Avg confidence: {summary['avg_confidence']:.1f}%" if summary['avg_confidence'] else "Avg confidence: N/A")

    return summary


def main() -> None:
    """Main execution - Parse Qwen VLM results and generate evaluation report."""
    parser = argparse.ArgumentParser(
        description="Parse Qwen VLM results and generate evaluation report."
    )
    parser.add_argument(
        "--vlm-results",
        type=Path,
        default=Path("vlm_outputs/qwen_pilot_results.csv"),
        help="Path to Qwen VLM results CSV (from run_qwen_pilot.py)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vlm_outputs"),
        help="Output directory for evaluation report"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("VLM EVALUATION")
    print("=" * 70)

    if not args.vlm_results.exists():
        print(f"\n Error: VLM results file not found: {args.vlm_results}")
        print(f"  Make sure you've run: python vlm/run_qwen_pilot.py")
        return

    print(f"\n[1/2] Loading VLM results from {args.vlm_results}...")
    vlm_df = load_vlm_results(args.vlm_results)
    print(f"    Loaded {len(vlm_df)} VLM evaluations")

    print(f"\n[2/2] Generating evaluation report...")
    report = generate_vlm_evaluation_report(vlm_df, args.output_dir)

    print(f"\nOutput saved to: {args.output_dir.absolute()}\n")


if __name__ == "__main__":
    main()