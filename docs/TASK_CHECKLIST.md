# Project task checklist (Clemson baseline + VLM)

Quick reference for what’s implemented in code vs what we still need to **run** or **write up**. Last updated Apr 2026.

---

## Week 1 — Reproduce baseline pipeline

| Item | Code / repo | Status |
|------|-------------|--------|
| Teacher = SAM, student = PIDNet, losses wired | `train.py`, `pidnet_utils/`, `models/` | Done |
| Dataset layout + loader | `datasets/segmentation_data.py`, `data/` | Done |
| Train + eval + infer entry points | `train.py`, `eval.py`, `infer.py` | Done |
| Smoke test (short run) | `--test-val`, `--max-batches`, `run_ablation.py` | Done |

---

## Week 2 — Baseline numbers, hard cases, VLM pilot

| Item | Code / repo | Status |
|------|-------------|--------|
| Metrics: mIoU, P/R, F1, Rand, speed | `eval.py`, `utils/experiment.py` | Done |
| Logging + checkpoints + run configs | `train.py`, `logs/*_config.json` | Done |
| Weekly results CSV template | `eval.py` → `weekly_results_template.csv` | Done |
| Reproducibility (seed, deterministic) | `utils/experiment.py` | Done |
| Hard-case folders + visual review | `hard_cases/` | Done |
| VLM pilot (Qwen), prompts | `vlm/run_qwen_pilot.py`, `vlm/prompt_templates.py` | Done |
| Big tables / figures for report | `baseline_eval.py` | Done (need real runs) |
| Compare our numbers to WACV paper | — | **Write-up** after we have final metrics |

---

## Week 3 — VLM-guided labels + ablations

| Item | Code / repo | Status |
|------|-------------|--------|
| Switch: SAM only / VLM / fused | `--label-mode` in `utils/get_args.py`, `train.py` | Done |
| VLM-filtered masks | `generate_vlm_masks.py` | Done |
| Fused masks + refinement rules | `generate_fused_masks.py`, `vlm/refinement.py` | Done |
| Candidate crops from teacher | `vlm/build_candidate_manifest.py` | Done |
| Ablation runner | `run_ablation.py`, `cluster/palmetto_final_ablation.slurm` | Done |

---

## Week 4–5 — Threshold calibration, final eval, report

| Item | Code / repo | Status |
|------|-------------|--------|
| Threshold sweep (train per combo) | `cluster/palmetto_threshold_sweep.slurm` | Done — **submit on Palmetto** |
| Pick best thresholds using **val** (not test) | Use val metrics from training logs; hold test for final | **Process** — code gives val loss/mIoU each epoch |
| Full eval on chosen checkpoint | `eval.py` + `--single-model` | Done |
| Qualitative: success / failure panels | `baseline_eval.py` | Done |
| Method write-up (prompts, rules, efficiency) | README + this repo; paper text | **Draft** in report |

### Still on us (not missing code, just work)

- [ ] Finish threshold jobs on the cluster and pick accept/reject/uncertain numbers with a short justification (val-based).
- [ ] Fill final tables (baseline vs VLM filter vs fused, hard cases if we subset-eval).
- [ ] Optional but good: pseudo-label precision/recall vs threshold on a **small labeled calibration set** — would need a tiny script if advisor wants that table explicitly; main segmentation metrics are already in `eval.py`.

### Dataset doc

- [ ] Clean up placeholders in `DATASET_DOCUMENTATION.md` when someone has time.

---

## One-line summary

**All the main pipeline code is in place** (train/eval, three label modes, fusion + refinement, VLM pilot, ablations, cluster scripts). What’s left is mostly **running experiments**, **picking thresholds from validation**, and **writing the 2-pager / figures** — not reinventing the codebase.

---

## Useful commands (copy-paste)

```bash
# Smoke ablation
python3 run_ablation.py --device cpu --epochs 1 --batch-size 1 --num-workers 0 --test-val --max-batches 2

# Cluster (after editing paths/modules for Palmetto)
sbatch cluster/palmetto_threshold_sweep.slurm
sbatch cluster/palmetto_final_ablation.slurm
```

Palmetto setup line-by-line: [`cluster/README.md`](../cluster/README.md).

If something breaks, check `logs/` and the saved `*_config.json` next to each experiment name.
