# Palmetto (SLURM) — how we run things

Paths and module names change over time; adjust if `module load` fails (check `module avail` on Palmetto).

## One-time setup (login node)

```bash
# From where you keep projects, e.g. scratch (replace with your path)
cd /path/to/your/scratch
git clone <your-repo-url> VLM-Wildfire-Detection
cd VLM-Wildfire-Detection

# Python env (3.10 matches the Slurm scripts)
module purge
module load python/3.10
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-vlm.txt   # only if you run Qwen on the cluster

# Layout the run expects (Slurm logs, VLM CSVs, etc.)
mkdir -p logs vlm_outputs checkpoints seg_outputs

# Dataset: copy or rsync your `data/` tree here — we do not commit full data.
# Imagenet PIDNet weights go under pretrained_models/ per README.
```

## Before threshold sweep or ablation

1. **VLM CSV for fusion** — `generate_fused_masks.py` defaults to `vlm_outputs/qwen_pilot_results.csv`. Generate that on Palmetto or copy it in from your laptop after a pilot run.
2. **ImageNet backbone** — ensure `pretrained_models/imagenet/PIDNet_S_ImageNet.pth.tar` (or M/L) exists so `train.py` can start.
3. **Partition / GPU** — edit the `#SBATCH` lines in the `.slurm` files if your group uses a different queue (e.g. `gpu`, `k40`, etc.).

## Walltime warning

`palmetto_threshold_sweep.slurm` runs **nine** (fused mask regen + train + eval) sequences **in one job**. Fifty epochs per train can exceed **24 hours** total. If the job dies mid-loop, either:

- raise `#SBATCH --time=...`, or  
- submit **one combo per job** (copy the inner body into a param script / array job), or  
- reduce `--epochs` for debugging.

## Submit jobs

From the repo root (same place as `train.py`):

```bash
module load python/3.10 cuda   # if needed for interactive check
source .venv/bin/activate
mkdir -p logs vlm_outputs

# Threshold study (edit .slurm first if needed)
sbatch cluster/palmetto_threshold_sweep.slurm

# Full baseline / VLM / fused ablation (after you like the thresholds in the script)
sbatch cluster/palmetto_final_ablation.slurm
```

Monitor:

```bash
squeue -u $USER
tail -f logs/slurm_threshold_<JOBID>.out
```

Artifacts: checkpoints under `checkpoints/`, metrics CSVs from `eval.py`, Slurm stdout/stderr under `logs/`.

## Quick interactive smoke test (optional)

```bash
salloc --partition=gpu --gres=gpu:1 --mem=48G --time=01:00:00
module load python/3.10 cuda
cd /path/to/VLM-Wildfire-Detection && source .venv/bin/activate
python3 run_ablation.py --device cuda --epochs 1 --batch-size 2 --num-workers 4 --test-val --max-batches 2 --deterministic
```

Exit the alloc when done.
