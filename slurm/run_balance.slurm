#!/bin/bash
# Balance batch (2026-10) under the gt-in-radius sparse schema, plus the
# human-teacher arms on transport.  Live config stack (no --base-config):
# neither tag has an older sweep to match, so each run's config.yaml is the
# frozen record.  One array task = one (scenario, tag, variant, seed group);
# the group's seeds share the GPU three at a time through paper_sweep --jobs 3,
# as in run_sweep.slurm / run_poster.slurm.
#
# Three phases, submitted IN ORDER (the radius is pinned between B and D):
#   B  sbatch --array=0     slurm/run_balance.slurm   # teacher audit + demo stats (minutes)
#      -> pin gt_radius in config/environments/balance.yaml, relabel the demos
#         (analysis/relabel_demos_sparse.py balance), commit, push, pull here.
#   D  sbatch --array=1-13  slurm/run_balance.slurm   # balance1: 7 arms
#   E  sbatch --array=14-20 slurm/run_balance.slurm   # transport_human1: 4 arms
#      sbatch --array=7     slurm/run_balance.slurm   # one task, e.g. a killed one
#
# Index -> task (scenario tag variant seeds resume-interval):
#    0  balance   balance1          audit                     -      -
#    1  balance   balance1          bc_eval                   0,1,2  20
#    2  balance   balance1          maddpg                    0,1,2  20
#    3  balance   balance1          maddpg                    3,4    20
#    4  balance   balance1          rlfd                      0,1,2  20
#    5  balance   balance1          rlfd                      3,4    20
#    6  balance   balance1          rft                       0,1,2  20
#    7  balance   balance1          rft                       3,4    20
#    8  balance   balance1          ibmarl                    0,1,2  20   (no gate, mixing)
#    9  balance   balance1          ibmarl                    3,4    20
#   10  balance   balance1          ibmarl_gated_a0.4         0,1,2  20   (IBMARL: gate + mixing)
#   11  balance   balance1          ibmarl_gated_a0.4         3,4    20
#   12  balance   balance1          ibmarl_strict_gated_a0.4  0,1,2  20   (gate, no mixing)
#   13  balance   balance1          ibmarl_strict_gated_a0.4  3,4    20
#   14  transport transport_human1  bc_eval_human             0,1,2  20
#   15  transport transport_human1  rlfd_human                0,1,2  20
#   16  transport transport_human1  rlfd_human                3,4    20
#   17  transport transport_human1  rft_human                 0,1,2  20
#   18  transport transport_human1  rft_human                 3,4    20
#   19  transport transport_human1  ibmarl_gated_a0.4_human   0,1,2  20
#   20  transport transport_human1  ibmarl_gated_a0.4_human   3,4    20
#
# The ibmarl runs log both protocols (eval_reward_mean = arbitrated RL+IL,
# rl_only_episode_reward_mean = RL actors alone), so "RL" and "RL+IL" are two
# columns of one run, not two tasks.  Copy back with
#   scp -r u1183156@notchpeak.chpc.utah.edu:~/code/IBMARL/results/{balance1,transport_human1} results/
#
# Cost: balance is 500 iterations of 3000 frames (300 sequential VMAS steps per
# collection, 500 optimiser steps, two 300-step eval protocols every 2nd
# iteration) -- roughly 45-60 s/iteration three to a GPU, so an IBMARL group may
# take 6-8 h and need one round of the walltime resubmit below; baselines should
# fit.  transport_human1 is transport800's 800-iteration budget at eval_interval
# 4 (~21 s/iteration alone), 6-9 h three to a GPU: expect one resubmit, as
# run_sweep.slurm records for transport.  Resume interval 20: each resume point
# dumps the 1M-frame buffer (~330-400 MB), so 5 would be mostly I/O.
#SBATCH --job-name=balance
#SBATCH --account=dbrown-gpu-np
#SBATCH --partition=dbrown-gpu-np
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-20
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1183156/logs/%x-%A_%a.out

set -euo pipefail

JOBS=${JOBS:-3}                      # concurrent runs per GPU
THREADS=${THREADS:-4}                # OMP threads per run (JOBS x THREADS <= cpus-per-task)
RUN_TIMEOUT=${RUN_TIMEOUT:-440m}     # --time minus margin for the last resume dump
timeout --signal=INT "$RUN_TIMEOUT" true || { echo "RUN_TIMEOUT=$RUN_TIMEOUT is not a duration GNU timeout accepts"; exit 1; }
REPO=${REPO:-$HOME/code/IBMARL}
SCRIPT="$REPO/slurm/run_balance.slurm"
AUDIT_RADII=${AUDIT_RADII:-0.3,0.5,0.75,1.05}
AUDIT_EPISODES=${AUDIT_EPISODES:-400}

# scenario tag variant seeds resume-interval
TASKS=(
  "balance   balance1         audit                    -     -"
  "balance   balance1         bc_eval                  0,1,2 20"
  "balance   balance1         maddpg                   0,1,2 20"
  "balance   balance1         maddpg                   3,4   20"
  "balance   balance1         rlfd                     0,1,2 20"
  "balance   balance1         rlfd                     3,4   20"
  "balance   balance1         rft                      0,1,2 20"
  "balance   balance1         rft                      3,4   20"
  "balance   balance1         ibmarl                   0,1,2 20"
  "balance   balance1         ibmarl                   3,4   20"
  "balance   balance1         ibmarl_gated_a0.4        0,1,2 20"
  "balance   balance1         ibmarl_gated_a0.4        3,4   20"
  "balance   balance1         ibmarl_strict_gated_a0.4 0,1,2 20"
  "balance   balance1         ibmarl_strict_gated_a0.4 3,4   20"
  "transport transport_human1 bc_eval_human            0,1,2 20"
  "transport transport_human1 rlfd_human               0,1,2 20"
  "transport transport_human1 rlfd_human               3,4   20"
  "transport transport_human1 rft_human                0,1,2 20"
  "transport transport_human1 rft_human                3,4   20"
  "transport transport_human1 ibmarl_gated_a0.4_human  0,1,2 20"
  "transport transport_human1 ibmarl_gated_a0.4_human  3,4   20"
)
[ "$SLURM_ARRAY_TASK_ID" -lt "${#TASKS[@]}" ] || { echo "index $SLURM_ARRAY_TASK_ID is past the table (0-$((${#TASKS[@]}-1)))"; exit 1; }
read -r SCENARIO TAG V S RESUME_INTERVAL <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> $SCENARIO/$TAG  $V seeds $S  ($JOBS per GPU, $THREADS threads each) on $(hostname)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ibmarl
cd "$REPO"

python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA on $(hostname)"; exit 1; }

if [ "$V" = audit ]; then
  # Phase B: the teacher's basin-reach per radius (and the demos' side of the
  # same table).  Nothing is trained; the decision rule is printed at the end.
  mkdir -p "results/$TAG/_teacher_audit"
  python -m analysis.relabel_demos_sparse "$SCENARIO" --radii "$AUDIT_RADII" --stats-only \
    | tee "results/$TAG/_teacher_audit/demo_stats.txt"
  python -m analysis.balance_teacher_audit --scenario "$SCENARIO" --episodes "$AUDIT_EPISODES" \
    --radii "$AUDIT_RADII" --csv "results/$TAG/_teacher_audit/audit.csv" \
    bundled=teachers/balance/policy_checkpoint.pth \
    | tee "results/$TAG/_teacher_audit/audit.txt"
  exit "${PIPESTATUS[0]}"
fi

# Pre-check: build the run config once and resolve its artifacts, so a missing
# overlay (e.g. rft_balance.yaml) or demo file fails here, not after the GPU
# has been held for an hour.  Writes only results/_precheck/.
python - "$V" "$SCENARIO" <<'PY' || { echo "pre-check failed for $SCENARIO/$V -- git pull? relabelled demos committed?"; exit 1; }
import sys
from analysis.paper_run import build_cfg
from src.util.paths import resolve_path
v, scenario = sys.argv[1:3]
cfg = build_cfg(v, scenario, 0, None, tag="_precheck")
for key in ("r2bc_checkpoint_path", "demonstrations_path"):
    if key in cfg:
        p = resolve_path(cfg[key])
        assert p.exists(), f"{key} -> {p} does not exist"
        print(f"[pre-check] {key}: {cfg[key]}")
print(f"[pre-check] {scenario}/{v}: n_iters {cfg.get('n_iters')}, total_frames {cfg.get('total_frames')}, "
      f"sparse_rewards {cfg.get('sparse_rewards')}, gt_radius {cfg.get('gt_radius')}")
PY

set +e
timeout --signal=INT "$RUN_TIMEOUT" python -m analysis.paper_sweep \
  --scenario "$SCENARIO" --tag "$TAG" --only "$V" --seeds "$S" \
  --jobs "$JOBS" --threads "$THREADS" --resume-interval "$RESUME_INTERVAL"
rc=$?
set -e
if [ $rc -eq 124 ]; then
  echo "walltime: stopped at $RUN_TIMEOUT, resubmitting index $SLURM_ARRAY_TASK_ID to continue from the group's resume points"
  # Resubmit to the account/partition this task actually ran under (a
  # command-line -A/-p override would otherwise be lost to the #SBATCH header).
  sbatch -J "$SLURM_JOB_NAME" -A "$SLURM_JOB_ACCOUNT" -p "$SLURM_JOB_PARTITION" \
    --export=ALL --array="$SLURM_ARRAY_TASK_ID" "$SCRIPT"
  exit 0
fi
exit $rc
