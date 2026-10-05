#!/bin/bash
# Balance batch (2026-10) under the gt-in-radius sparse schema, plus the
# human-teacher arms on transport.  Live config stack (no --base-config):
# neither tag has an older sweep to match, so each run's config.yaml is the
# frozen record.  One array task = one (scenario, tag, variant, seed group);
# the group's seeds share the GPU three at a time through paper_sweep --jobs 3,
# as in run_sweep.slurm / run_poster.slurm.
#
# One submission runs everything (the radius was chosen a priori, 0.5, see
# config/environments/balance.yaml; the audit in task 0 reports the teacher's
# basin-reach grid for the record and gates nothing):
#      sbatch slurm/run_balance.slurm                  # all 21 tasks
#      sbatch --array=1-13  slurm/run_balance.slurm    # balance1 only
#      sbatch --array=14-20 slurm/run_balance.slurm    # transport_human1 only
#      sbatch --array=7     slurm/run_balance.slurm    # one task, e.g. a killed one
# Every task first runs analysis/ensure_demo_artifacts.py under flock, which
# generates the two derived demo files that need torch to make
# (teachers/balance/demonstrations_sparse.pt at the yaml's gt_radius, and
# teachers/human_transport_90_demos/demonstrations_legacy.pt) if they are not
# on disk, then checks the run's config and artifacts before the GPU is held.
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
  # Task 0: the teacher's basin-reach per radius (and the demos' side of the
  # same table), for the record.  Nothing is trained; the 0.4-0.6 band check
  # is printed at the end so a too-strong teacher is visible early.
  mkdir -p "results/$TAG/_teacher_audit"
  python -m analysis.relabel_demos_sparse "$SCENARIO" --radii "$AUDIT_RADII" --stats-only \
    | tee "results/$TAG/_teacher_audit/demo_stats.txt"
  python -m analysis.balance_teacher_audit --scenario "$SCENARIO" --episodes "$AUDIT_EPISODES" \
    --radii "$AUDIT_RADII" --csv "results/$TAG/_teacher_audit/audit.csv" \
    bundled=teachers/balance/policy_checkpoint.pth \
    | tee "results/$TAG/_teacher_audit/audit.txt"
  exit "${PIPESTATUS[0]}"
fi

# Pre-flight (see analysis/ensure_demo_artifacts.py): generate the derived demo
# files if missing -- serialised across concurrently starting tasks by flock on
# a repo-local lock -- then build the run config and resolve its artifacts, so a
# missing overlay or file fails here, not after the GPU has been held.
mkdir -p results/_precheck
flock --timeout 1800 results/_precheck/artifacts.lock \
  python -m analysis.ensure_demo_artifacts "$V" "$SCENARIO" \
  || { echo "pre-flight failed for $SCENARIO/$V -- git pull? see analysis/ensure_demo_artifacts.py"; exit 1; }

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
