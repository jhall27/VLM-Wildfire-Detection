import json
import os
import random
import time
from typing import Dict, Iterable

import numpy as np
import torch


def ensure_dirs(paths: Iterable[str]) -> None:
    # Small helper so train/eval do not fail if folders are missing.
    for path in paths:
        os.makedirs(path, exist_ok=True)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    # Keep runs as consistent as possible across repeated tests.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def configure_torch_runtime(disable_cudnn: bool = False) -> None:
    # Some cluster nodes expose CUDA but fail when cuDNN initializes.
    if disable_cudnn:
        torch.backends.cudnn.enabled = False


def save_run_config(args, output_path: str) -> None:
    # Save the exact arguments used for a run so it is easy to repeat later.
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)


def measure_inference_speed(model, sample_batch, device: str, warmup: int = 5, steps: int = 20) -> Dict[str, float]:
    # Quick speed check used during evaluation.
    model.eval()
    inputs = sample_batch["img"].to(device)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model(inputs, labels=None, bd_gt=None, plot_outputs=True)

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(steps):
            model(inputs, labels=None, bd_gt=None, plot_outputs=True)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    total_images = max(1, inputs.shape[0] * steps)
    seconds_per_image = elapsed / total_images
    return {
        "seconds_per_image": seconds_per_image,
        "milliseconds_per_image": seconds_per_image * 1000.0,
        "fps": 1.0 / seconds_per_image if seconds_per_image > 0 else 0.0,
    }


def get_metric_template() -> Dict[str, str]:
    # Template for the weekly results table.
    return {
        "week": "",
        "experiment": "",
        "dataset_split": "",
        "teacher_model": "",
        "student_model": "",
        "epochs": "",
        "mIoU": "",
        "precision": "",
        "recall": "",
        "f1": "",
        "milliseconds_per_image": "",
        "fps": "",
        "checkpoint": "",
        "notes": "",
    }
