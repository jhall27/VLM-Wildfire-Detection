# VLM Wildfire Detection Baseline

This repo is our cleaned baseline for the paper *Detecting Wildfires on UAVs with Real-time Segmentation Trained by Larger Teacher Models* (WACV 2025).

The main point of this version is to have a working starting codebase that we can build on later for our own project ideas, especially the VLM part.

## What This Repo Does

The baseline uses a teacher-student setup:

- Teacher model: SAM
- Student model: PIDNet

The rough idea is:

1. use bounding boxes from the dataset
2. turn those boxes into pseudo-label masks with SAM
3. train PIDNet on those masks
4. evaluate against manual masks when they are available

## Main Files

- `train.py`: trains the segmentation model
- `eval.py`: evaluates a checkpoint and writes metrics
- `infer.py`: runs inference on one image and saves a predicted mask
- `generate_pseudo_labels.py`: generates SAM masks from box labels
- `wildfire_dataset_loader.py`: simpler loader for dataset checking
- `vlm/build_manifest.py`: builds a small hard-case manifest for VLM tests
- `vlm/run_qwen_pilot.py`: runs or stages the Qwen pilot prompts
- `vlm/prompt_templates.py`: stores the prompt styles we compare

## Dataset Layout

The current checked-in dataset snapshot is under `data/`:

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

Current snapshot summary:

- train: 2068 images
- valid: 453 images
- test: 40 images
- total: 2561 images

Formats in the current repo snapshot:

- images: `.jpeg`
- labels: `.xml`
- SAM masks: `.png`
- manual masks: `.png`

One thing to keep in mind is that this looks like a reduced snapshot of the bigger combined dataset, not the full final version of everything.

## Environment Setup

The original project was built around Python 3.8, so that is still the safest version to use if possible.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional, only if you want to regenerate masks with SAM:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Quick Checks

### Small training/validation check

```bash
python3 train.py --device cpu --epochs 1 --batch-size 1 --num-workers 0 --test-val --max-batches 2 --deterministic
```

This is just a quick smoke test to make sure the baseline pipeline runs.

### Small evaluation check

```bash
python3 eval.py --device cpu --single-model checkpoints/sam_sup_pidnet_s.pt --model-size s --num-workers 0 --max-batches 2 --deterministic
```

This writes:

- `eval_results.csv`
- `weekly_results_template.csv`

### Single-image inference

```bash
python3 infer.py --device cpu --model-size s --exp sam_sup_pidnet_s --input-image data/images/test/15_2161.jpeg
```

The output mask gets saved under `seg_outputs/`.

## Training

Default:

```bash
python3 train.py
```

Example:

```bash
python3 train.py \
  --data-dir data \
  --batch-size 4 \
  --epochs 50 \
  --device cuda \
  --exp sam_sup_pidnet_s \
  --model-size s \
  --seed 42 \
  --deterministic
```

Training saves:

- log CSVs in `logs/`
- config snapshots in `logs/`
- checkpoints in `checkpoints/`

## Evaluation

Example:

```bash
python3 eval.py \
  --device cpu \
  --single-model checkpoints/sam_sup_pidnet_s.pt \
  --model-size s \
  --seed 42 \
  --deterministic
```

Current evaluation metrics include:

- mIoU
- precision
- recall
- F1
- Rand index / binary accuracy
- inference speed
- FPS

## Part 3 VLM Pilot

The repo now has a small VLM pilot workflow built around the `hard_cases/` folder.

The three prompt styles are:

- `yes_no`
- `confidence_score`
- `region_reasoning`

Build the hard-case manifest:

```bash
python3 vlm/build_manifest.py
```

Create a dry-run results sheet:

```bash
python3 vlm/run_qwen_pilot.py --mode dry-run --prompt-style all
```

If Qwen and `transformers` are available locally, the same script can run the real pilot:

```bash
pip install -r requirements-vlm.txt
python3 vlm/run_qwen_pilot.py --mode local --prompt-style all --device cpu
```

Dry-run and real-run outputs both go under `vlm_outputs/`.

## Notes

- This repo is meant to be a clean baseline, not the final finished research system.
- The current checked-in dataset does not keep explicit source tags for Boreal vs AI For Mankind samples.
- Full paper-level reproduction would still require larger runs than the small smoke tests we used to verify the pipeline.
