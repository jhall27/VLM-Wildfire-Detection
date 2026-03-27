# Wildfire Dataset Documentation

## Overview

This file is just a summary of the dataset snapshot that is currently checked into the repo.

For this project, `data/` is the main source of truth. I also updated `dataset_summary.json` so it matches the current checked-in files.

One important note is that this looks like a smaller snapshot of the full combined wildfire dataset, not the complete final version.

## Intended Dataset Sources

The project is meant to use data from:

- AI For Mankind wildfire smoke data
- Boreal Forest Fire Subset-C

In the current repo, though, the data looks like a reduced XML-based subset. The original Boreal folder structure is not fully present in the form described in the paper/release instructions.

## Current Checked-In Dataset

Current layout:

```text
data/
├── images/
│   ├── train
│   ├── valid
│   └── test
├── labels/
│   ├── train
│   ├── valid
│   └── test
├── sam_masks/
│   ├── train
│   ├── valid
│   └── test
└── manual_masks/
    └── test
```

Current split counts:

| Split | Images | Labels | SAM Masks | Manual Masks |
|-------|--------|--------|-----------|--------------|
| Train | 2068 | 2068 | 2068 | 0 |
| Valid | 453 | 453 | 453 | 0 |
| Test | 40 | 40 | 40 | 40 |
| Total | 2561 | 2561 | 2561 | 40 |

So right now:

- train and valid use SAM masks
- manual masks are only present for test

## File Formats

Observed in the current snapshot:

- image format: `.jpeg`
- label format: `.xml`
- SAM mask format: `.png`
- manual mask format: `.png`

Images I checked were RGB and mostly `640 x 480`.

The training code resizes images and masks to `1080 x 1920`.

Example filenames:

- image: `10_117.jpeg`
- label: `10_117.xml`
- SAM mask: `10_117.png`

## How The Labels Are Used

The current repo uses:

- XML bounding boxes in `data/labels/*`
- SAM-generated masks in `data/sam_masks/*`
- manual masks in `data/manual_masks/test`

So the flow is:

1. start with bounding boxes
2. generate pseudo-label masks with SAM
3. train PIDNet on those masks
4. evaluate on manual masks where available

Only XML labels are checked in right now, even though the code still supports TXT labels for other dataset versions.

## Split Usage

### Train

- purpose: model training
- supervision: SAM masks
- count: 2068

### Valid

- purpose: validation during training
- supervision: SAM masks
- count: 453

### Test

- purpose: final evaluation
- supervision available: SAM masks + manual masks
- count: 40

## Data Quality Notes

Things that seem okay:

- image, label, and SAM-mask counts match in the current snapshot
- sample images open correctly
- the current summary file now matches the data that is actually in the repo

Things that are still limited:

- only the test split has manual masks
- source separation is not preserved clearly, so it is hard to tell which samples came from Boreal vs AI For Mankind
- weak supervision means the SAM masks can still be noisy, especially around boundaries or hard smoke cases

## Preprocessing In The Baseline

From the current training loader:

- images are loaded as RGB
- training uses augmentations like crop, flip, rotation, perspective, blur, grayscale, invert, sharpness, and color jitter
- images and masks are resized to `1080 x 1920`
- images are normalized with ImageNet mean/std
- masks are converted into binary masks
- boundary maps are built from masks using Canny edges and dilation

## What Was Verified

Checked in the current repo:

- `data/images/{train,valid,test}` exists
- `data/labels/{train,valid,test}` exists
- `data/sam_masks/{train,valid,test}` exists
- `data/manual_masks/test` exists
- file counts line up for the checked snapshot
- sample images load as RGB and look consistent

Not fully checked in detail:

- full corruption scan of every file
- per-sample source mapping
- full visual inspection of every generated mask

## Suggested Next Cleanup Steps

If we want to improve the dataset side later, the next useful steps would be:

- add a manifest showing whether each sample is from Boreal or AI For Mankind
- add a script to verify exact one-to-one filename matching across images, labels, SAM masks, and manual masks
- export a few sample visual checks showing image, box, SAM mask, and manual mask together
- sync the full intended combined dataset if that is needed for later experiments

## Basic Reproduction Workflow

```bash
python3 train.py --device cpu --epochs 1 --batch-size 1 --num-workers 0 --test-val --max-batches 2 --deterministic
python3 eval.py --device cpu --single-model checkpoints/sam_sup_pidnet_s.pt --model-size s --num-workers 0 --max-batches 2 --deterministic
python3 wildfire_dataset_loader.py
```

## Bottom Line

The dataset setup in this repo is usable for the baseline pipeline, and the docs now match the actual checked-in snapshot. The main remaining limitations are that the dataset snapshot is reduced and that the original source split is not clearly preserved per sample.
