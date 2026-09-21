#!/bin/bash
# One array task = one (sweep, variant, seed group); the group's seeds share the
# task's GPU, three at a time, through paper_sweep --jobs 3.
#
#   sbatch slurm/run_sweep.slurm                # everything: buzzwire5 + transport800's remainder
#   sbatch --array=0-15 slurm/run_sweep.slurm   # buzzwire5 only
#   sbatch --array=16-31 slurm/run_sweep.slurm  # transport800 only
#   sbatch --array=10 slurm/run_sweep.slurm     # one task, e.g. to re-run a killed one
#
# Index -> task: 16 per sweep = 8 methods x 2 seed groups (even index = seeds
# 0,1,2; odd = seeds 3,4).  sweep = idx / 16, method = (idx % 16) / 2:
#    0-15   buzz_wire  buzzwire5     bc_eval 0,1  maddpg 2,3  rlfd 4,5  rft 6,7
#                                    ibmarl 8,9  ibmarl_gated_a0.4 10,11
#                                    ibmarl_strict_gated_a0.4 12,13  ..._1critic 14,15
#   16-31   transport  transport800  same order, +16
#
# Why groups share a GPU: a run holds ~350 MiB, so memory is no limit, and a
# 2080 ran three buzz-wire runs at ~2.2 h each in buzzwire4 -- close to a run
# alone -- so packing three is close to 3x cheaper in GPU-hours.  Each child
# gets OMP_NUM_THREADS=4 from paper_sweep (--cpus-per-task 12 covers three);
# an uncapped run spawns a thread per core and three of them thrash.
#
# Guards, so the array is safe to submit whole and to resubmit:
#   * paper_sweep skips every seed whose status.json says ok and resumes a
#     partial one from its checkpoints/resume, so a task over a finished group
#     exits in seconds.  transport800's six runs finished on the workstation
#     (bc_eval 0-2 = task 16, ibmarl_gated_a0.4 0-2 = task 26) are skipped ONLY
#     if results/transport800 has been rsynced to the cluster; otherwise leave
#     those two out:   sbatch --array=0-15,17-25,27-31 slurm/run_sweep.slurm
#   * at the walltime, `timeout` sends SIGINT with margin for the last resume
#     point; paper_sweep stops its children and the task resubmits ITSELF with
#     the same index, continuing where every seed left off.  A transport group
#     (~800 iterations sharing a GPU three ways) may need one round of this;
#     a buzz-wire group should not.
#
# SCENARIO / TAG / RESUME_INTERVAL may be forced via --export for an ad-hoc
# run; they override the table for every task of that submission.
#SBATCH --job-name=sweep
#SBATCH --account=dbrown-gpu-np
#SBATCH --partition=dbrown-gpu-np
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-31
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1183156/logs/%x-%A_%a.out

set -euo pipefail

JOBS=${JOBS:-3}                      # concurrent runs per GPU
THREADS=${THREADS:-4}                # OMP threads per run (JOBS x THREADS <= cpus-per-task)
# Wall clock handed to the group before it is stopped for resubmission: the
# --time above minus margin for the last resume dump and teardown.  GNU
# timeout takes ONE number with ONE unit suffix (minutes here).
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
SEED_GROUPS=("0,1,2" "3,4")

PER_SWEEP=$(( ${#METHODS[@]} * ${#SEED_GROUPS[@]} ))
sweep_idx=$(( SLURM_ARRAY_TASK_ID / PER_SWEEP ))
local_idx=$(( SLURM_ARRAY_TASK_ID % PER_SWEEP ))
[ "$sweep_idx" -lt "${#SWEEPS[@]}" ] || { echo "index $SLURM_ARRAY_TASK_ID is past the sweep table"; exit 1; }
read -r T_SCENARIO T_TAG T_RESUME <<< "${SWEEPS[$sweep_idx]}"
SCENARIO=${SCENARIO:-$T_SCENARIO}
TAG=${TAG:-$T_TAG}
RESUME_INTERVAL=${RESUME_INTERVAL:-$T_RESUME}
V=${METHODS[$((local_idx / ${#SEED_GROUPS[@]}))]}
S=${SEED_GROUPS[$((local_idx % ${#SEED_GROUPS[@]}))]}
echo "task $SLURM_ARRAY_TASK_ID -> $SCENARIO/$TAG  $V seeds $S  ($JOBS per GPU, $THREADS threads each) on $(hostname)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ibmarl
cd "$REPO"

python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA on $(hostname)"; exit 1; }

set +e
timeout --signal=INT "$RUN_TIMEOUT" python -m analysis.paper_sweep \
  --scenario "$SCENARIO" --tag "$TAG" --only "$V" --seeds "$S" \
  --jobs "$JOBS" --threads "$THREADS" --resume-interval "$RESUME_INTERVAL"
rc=$?
set -e
if [ $rc -eq 124 ]; then
  echo "walltime: stopped at $RUN_TIMEOUT, resubmitting index $SLURM_ARRAY_TASK_ID to continue from the group's resume points"
  sbatch -J "$SLURM_JOB_NAME" --export=ALL --array="$SLURM_ARRAY_TASK_ID" "$SCRIPT"
  exit 0
fi
exit $rc
