#!/bin/bash
#SBATCH --job-name=vlm-final-ablation
#SBATCH --output=logs/slurm_ablation_%j.out
#SBATCH --error=logs/slurm_ablation_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs vlm_outputs checkpoints

module purge
module load python/3.10
module load cuda
source .venv/bin/activate

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

# Update the thresholds below after the calibration sweep picks the best values.
python3 run_ablation.py \
  --device cuda \
  --epochs 50 \
  --batch-size 4 \
  --num-workers 8 \
  --model-size s \
  --seed 42 \
  --deterministic \
  --accept-threshold 70 \
  --reject-threshold 40 \
  --region-accept-threshold 60 \
  --uncertain-low 40 \
  --uncertain-high 70
