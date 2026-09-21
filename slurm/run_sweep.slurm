#!/bin/bash
# One (variant, seed) run per array task, over a TABLE of sweeps.
#
#   sbatch slurm/run_sweep.slurm                # everything: buzzwire5 + transport800's remainder
#   sbatch --array=0-39 slurm/run_sweep.slurm   # buzzwire5 only
#   sbatch --array=40-79 slurm/run_sweep.slurm  # transport800 only
#   sbatch --array=5,12 slurm/run_sweep.slurm   # re-run two specific tasks
#
# The array index walks SWEEPS below, 40 tasks per sweep (8 methods x 5 seeds):
#   idx -> sweep = idx / 40; within it variant = METHODS[(idx % 40) / 5], seed = idx % 5
#     0-39   buzz_wire  buzzwire5      (bc_eval 0-4, maddpg 5-9, rlfd 10-14, rft 15-19,
#                                       ibmarl 20-24, ibmarl_gated_a0.4 25-29,
#                                       ibmarl_strict_gated_a0.4 30-34, ..._1critic 35-39)
#    40-79   transport  transport800   (same order, +40)
#
# Two guards make the whole array safe to submit as-is and to resubmit:
#   * a task whose status.json already says ok exits immediately.  The six
#     transport800 runs finished on the workstation (bc_eval 40-42,
#     ibmarl_gated_a0.4 65-67) are skipped ONLY if results/transport800 has
#     been rsynced to the cluster first (which also puts every run in one
#     place for the figures); otherwise leave them out of the array:
#       sbatch --array=0-39,43-64,68-79 slurm/run_sweep.slurm
#   * a task that reaches the walltime is stopped by `timeout` with margin for
#     its last resume point, then resubmits ITSELF with the same index, and
#     paper_run continues from results/<tag>/<variant>/seed_N/checkpoints/resume.
#     A transport run (~800 iterations at ~30 s each alone on a 2080) needs
#     this; a buzz-wire run should not.
#
# One run per task because the 8 h cap does not fit several seeds sharing a GPU
# (~7-8 h per buzz-wire IBMARL run when three share a 2080), and because
# results then land in the canonical results/<tag>/<variant>/seed_N layout that
# analysis/paper3_figures.py reads directly.
#
# SCENARIO / TAG / RESUME_INTERVAL may still be forced via --export for an
# ad-hoc run; they override the table entry for every task of that submission.
#SBATCH --job-name=sweep
#SBATCH --account=dbrown-gpu-np
#SBATCH --partition=dbrown-gpu-np
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-79
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1183156/logs/%x-%A_%a.out

set -euo pipefail

SEEDS=${SEEDS:-5}
# Wall clock handed to the run before it is stopped for resubmission: the
# --time above minus margin for the final resume dump and teardown.  GNU
# timeout takes ONE number with ONE unit suffix -- "7h20m" is rejected with
# exit 125 before the run starts (that failed an entire 80-task array on
# 2026-09-21), so this is minutes.
RUN_TIMEOUT=${RUN_TIMEOUT:-440m}
timeout --signal=INT "$RUN_TIMEOUT" true || { echo "RUN_TIMEOUT=$RUN_TIMEOUT is not a duration GNU timeout accepts"; exit 1; }
REPO=${REPO:-$HOME/code/IBMARL}
SCRIPT="$REPO/slurm/run_sweep.slurm"

# scenario  tag  resume-interval.  Iterations between resume points: each one
# dumps the replay buffer (~330 MB for transport's 1M-frame buffer, so every 5
# iterations there would be mostly I/O; 20 is ~10 min of lost work at most).
SWEEPS=(
  "buzz_wire buzzwire5 5"
  "transport transport800 20"
)
METHODS=(bc_eval maddpg rlfd rft ibmarl ibmarl_gated_a0.4 ibmarl_strict_gated_a0.4 ibmarl_gated_a0.4_1critic)
PER_SWEEP=$(( ${#METHODS[@]} * SEEDS ))
sweep_idx=$(( SLURM_ARRAY_TASK_ID / PER_SWEEP ))
local_idx=$(( SLURM_ARRAY_TASK_ID % PER_SWEEP ))
[ "$sweep_idx" -lt "${#SWEEPS[@]}" ] || { echo "index $SLURM_ARRAY_TASK_ID is past the sweep table"; exit 1; }
read -r T_SCENARIO T_TAG T_RESUME <<< "${SWEEPS[$sweep_idx]}"
SCENARIO=${SCENARIO:-$T_SCENARIO}
TAG=${TAG:-$T_TAG}
RESUME_INTERVAL=${RESUME_INTERVAL:-$T_RESUME}
V=${METHODS[$((local_idx / SEEDS))]}
S=$((local_idx % SEEDS))
echo "task $SLURM_ARRAY_TASK_ID -> $SCENARIO/$TAG  $V seed $S  on $(hostname)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ibmarl
cd "$REPO"

STATUS="results/$TAG/$V/seed_$S/status.json"
if [ -f "$STATUS" ] && python -c "import json,sys; sys.exit(0 if json.load(open('$STATUS')).get('ok') else 1)"; then
  echo "already done ($STATUS says ok) -- nothing to do"; exit 0
fi

python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA on $(hostname)"; exit 1; }
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

set +e
timeout --signal=INT "$RUN_TIMEOUT" python -m analysis.paper_run \
  --variant "$V" --seed "$S" --scenario "$SCENARIO" --tag "$TAG" \
  --resume-interval "$RESUME_INTERVAL"
rc=$?
set -e
if [ $rc -eq 124 ]; then
  echo "walltime: stopped at $RUN_TIMEOUT, resubmitting index $SLURM_ARRAY_TASK_ID to continue from its resume point"
  sbatch -J "$SLURM_JOB_NAME" --export=ALL --array="$SLURM_ARRAY_TASK_ID" "$SCRIPT"
  exit 0
fi
exit $rc
