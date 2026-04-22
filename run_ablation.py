"""Run baseline / VLM / fused ablation experiments with consistent naming."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def resolve_device(requested_device: str) -> str:
    """Use MPS only when it is actually available, otherwise fall back cleanly."""
    if requested_device != "mps":
        return requested_device

    try:
        import torch
    except ImportError:
        print("PyTorch not available for MPS check, falling back to cpu.")
        return "cpu"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    print("MPS requested but not available on this system, falling back to cpu.")
    return "cpu"


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def refinement_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "generate_fused_masks.py",
        "--accept-threshold",
        str(args.accept_threshold),
        "--reject-threshold",
        str(args.reject_threshold),
        "--region-accept-threshold",
        str(args.region_accept_threshold),
        "--uncertain-low",
        str(args.uncertain_low),
        "--uncertain-high",
        str(args.uncertain_high),
        "--min-component-area",
        str(args.min_component_area),
    ]
    return cmd


def train_command(args: argparse.Namespace, label_mode: str, exp_name: str) -> list[str]:
    cmd = [
        sys.executable,
        "train.py",
        "--label-mode",
        label_mode,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--model-size",
        args.model_size,
        "--exp",
        exp_name,
        "--seed",
        str(args.seed),
    ]
    if args.max_batches:
        cmd.extend(["--max-batches", str(args.max_batches)])
    if args.deterministic:
        cmd.append("--deterministic")
    if args.disable_cudnn:
        cmd.append("--disable-cudnn")
    if args.test_val:
        cmd.append("--test-val")
    return cmd


def eval_command(args: argparse.Namespace, checkpoint_path: Path, label: str) -> list[str]:
    cmd = [
        sys.executable,
        "eval.py",
        "--device",
        args.device,
        "--single-model",
        str(checkpoint_path),
        "--model-size",
        args.model_size,
        "--num-workers",
        str(args.num_workers),
        "--metrics-output",
        f"eval_results_{label}.csv",
        "--results-template",
        f"weekly_results_{label}.csv",
        "--seed",
        str(args.seed),
    ]
    if args.max_batches:
        cmd.extend(["--max-batches", str(args.max_batches)])
    if args.deterministic:
        cmd.append("--deterministic")
    if args.disable_cudnn:
        cmd.append("--disable-cudnn")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three main ablation settings.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-size", default="s")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--test-val", action="store_true")
    parser.add_argument("--skip-refinement-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accept-threshold", type=int, default=70)
    parser.add_argument("--reject-threshold", type=int, default=40)
    parser.add_argument("--region-accept-threshold", type=int, default=60)
    parser.add_argument("--uncertain-low", type=int, default=40)
    parser.add_argument("--uncertain-high", type=int, default=70)
    parser.add_argument("--min-component-area", type=int, default=120)
    parser.add_argument(
        "--eval-checkpoint",
        choices=["miou", "loss"],
        default="miou",
        help="Which saved checkpoint to evaluate after training. Default=miou for consistency with threshold-sweep runs.",
    )
    parser.add_argument("--modes", nargs="+", default=["sam", "vlm", "fused"])
    # Backward-friendly aliases for common CLI mistakes.
    parser.add_argument("--mode", choices=["sam", "vlm", "fused"], default=None)
    parser.add_argument("--exp", default="ablation")
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    if args.mode:
        args.modes = [args.mode]

    if not args.skip_refinement_build:
        run_command([sys.executable, "generate_vlm_masks.py"], args.dry_run)
        run_command(refinement_command(args), args.dry_run)

    modes = [
        ("sam", "baseline"),
        ("vlm", "vlm_filter"),
        ("fused", "vlm_fused"),
    ]

    for label_mode, label in modes:
        if label_mode not in args.modes:
            continue
        exp_name = f"{args.exp}_{label}_{args.model_size}"
        run_command(train_command(args, label_mode, exp_name), args.dry_run)
        suffix = "_miou.pt" if args.eval_checkpoint == "miou" else ".pt"
        checkpoint_path = Path("checkpoints") / f"{exp_name}{suffix}"
        run_command(eval_command(args, checkpoint_path, label), args.dry_run)


if __name__ == "__main__":
    main()
