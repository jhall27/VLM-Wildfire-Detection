#!/bin/bash
#SBATCH --job-name=vlm-threshold-sweep
#SBATCH --output=logs/slurm_threshold_%j.out
#SBATCH --error=logs/slurm_threshold_%j.err
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

# Update these for your Palmetto environment.
module purge
module load python/3.10
module load cuda

# Either activate your env here, or comment this out if modules already include
# the correct packages.
source .venv/bin/activate

export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl"
mkdir -p "$MPLCONFIGDIR"

for ACCEPT in 65 70 75; do
  for REJECT in 30 40 50; do
    if [ "$REJECT" -ge "$ACCEPT" ]; then
      continue
    fi

    TAG="a${ACCEPT}_r${REJECT}"

    python3 generate_fused_masks.py \
      --accept-threshold "$ACCEPT" \
      --reject-threshold "$REJECT" \
      --region-accept-threshold 60 \
      --uncertain-low "$REJECT" \
      --uncertain-high "$ACCEPT" \
      --output-csv "vlm_outputs/fused_refinement_decisions_${TAG}.csv"

    python3 train.py \
      --label-mode fused \
      --device cuda \
      --epochs 50 \
      --batch-size 4 \
      --num-workers 8 \
      --model-size s \
      --exp "threshold_${TAG}" \
      --seed 42 \
      --deterministic

    python3 eval.py \
      --device cuda \
      --single-model "checkpoints/threshold_${TAG}_miou.pt" \
      --model-size s \
      --num-workers 8 \
      --metrics-output "eval_results_${TAG}.csv" \
      --results-template "weekly_results_${TAG}.csv" \
      --seed 42 \
      --deterministic
  done
done
