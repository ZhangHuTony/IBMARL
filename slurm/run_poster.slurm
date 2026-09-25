#!/bin/bash
# Poster batch (2026-09-24): the arms that never ran under the LEGACY -1/step
# schema, built from the frozen bases in config/legacy/ (--base-config legacy)
# so they are comparable with the sweeps already on disk (paper3/paper4 for
# navigation, buzzwire3/buzzwire3_reg for buzz-wire).  One array task = one
# (scenario, tag, variant, seed group); the group's seeds share the GPU three
# at a time through paper_sweep --jobs 3, as in run_sweep.slurm.
#
#   sbatch slurm/run_poster.slurm              # all 13 tasks
#   sbatch --array=0-4 slurm/run_poster.slurm  # navigation only
#   sbatch --array=5-12 slurm/run_poster.slurm # buzz-wire only
#   sbatch --array=7 slurm/run_poster.slurm    # one task, e.g. a killed one
#
# Index -> task (scenario tag variant seeds resume-interval):
#    0  navigation paper5     ibmarl_gated_a0.4         0,1,2
#    1  navigation paper5     ibmarl_gated_a0.4         3,4
#    2  navigation paper5     ibmarl_strict_gated_a0.4  0,1,2
#    3  navigation paper5     ibmarl_strict_gated_a0.4  3,4
#    4  navigation paper5     ibmarl                    4      (paper4's seed 4 died at 131/300)
#    5  buzz_wire  buzzwire6  ibmarl_gated_a0.4         0,1,2
#    6  buzz_wire  buzzwire6  ibmarl_gated_a0.4         3,4
#    7  buzz_wire  buzzwire6  ibmarl_gated_a0.4_human   0,1,2
#    8  buzz_wire  buzzwire6  ibmarl_gated_a0.4_human   3,4
#    9  buzz_wire  buzzwire6  rft_human                 0,1,2
#   10  buzz_wire  buzzwire6  rft_human                 3,4
#   11  buzz_wire  buzzwire6  rlfd_human                0,1,2
#   12  buzz_wire  buzzwire6  rlfd_human                3,4
#
# The reused arms of paper5/buzzwire6 (baselines, teacher, the no-gate and
# no-mixing IBMARL arms) are symlinks on the workstation, not runs, so nothing
# here collides with them and the cluster's results/<tag> holds only the new
# runs.  Copy back with
#   scp -r u1183156@notchpeak.chpc.utah.edu:~/code/IBMARL/results/{paper5,buzzwire6} results/
#
# Cost: a navigation run is ~1.5-2 h and a buzz-wire run ~2-2.5 h at three per
# GPU (paper3, buzzwire3 wall_seconds), so every group ends well inside the
# 8 h walltime; the timeout/resubmit guard is kept for safety only.
#SBATCH --job-name=poster
#SBATCH --account=dbrown-gpu-np
#SBATCH --partition=dbrown-gpu-np
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --array=0-12
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1183156/logs/%x-%A_%a.out

set -euo pipefail

JOBS=${JOBS:-3}                      # concurrent runs per GPU
THREADS=${THREADS:-4}                # OMP threads per run (JOBS x THREADS <= cpus-per-task)
RUN_TIMEOUT=${RUN_TIMEOUT:-440m}     # --time minus margin for the last resume dump
timeout --signal=INT "$RUN_TIMEOUT" true || { echo "RUN_TIMEOUT=$RUN_TIMEOUT is not a duration GNU timeout accepts"; exit 1; }
REPO=${REPO:-$HOME/code/IBMARL}
SCRIPT="$REPO/slurm/run_poster.slurm"
BASE_CONFIG=${BASE_CONFIG:-legacy}

# scenario tag variant seeds resume-interval
TASKS=(
  "navigation paper5    ibmarl_gated_a0.4        0,1,2 25"
  "navigation paper5    ibmarl_gated_a0.4        3,4   25"
  "navigation paper5    ibmarl_strict_gated_a0.4 0,1,2 25"
  "navigation paper5    ibmarl_strict_gated_a0.4 3,4   25"
  "navigation paper5    ibmarl                   4     25"
  "buzz_wire  buzzwire6 ibmarl_gated_a0.4        0,1,2 5"
  "buzz_wire  buzzwire6 ibmarl_gated_a0.4        3,4   5"
  "buzz_wire  buzzwire6 ibmarl_gated_a0.4_human  0,1,2 5"
  "buzz_wire  buzzwire6 ibmarl_gated_a0.4_human  3,4   5"
  "buzz_wire  buzzwire6 rft_human                0,1,2 5"
  "buzz_wire  buzzwire6 rft_human                3,4   5"
  "buzz_wire  buzzwire6 rlfd_human               0,1,2 5"
  "buzz_wire  buzzwire6 rlfd_human               3,4   5"
)
[ "$SLURM_ARRAY_TASK_ID" -lt "${#TASKS[@]}" ] || { echo "index $SLURM_ARRAY_TASK_ID is past the table (0-$((${#TASKS[@]}-1)))"; exit 1; }
read -r SCENARIO TAG V S RESUME_INTERVAL <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> $SCENARIO/$TAG  $V seeds $S  base-config $BASE_CONFIG  ($JOBS per GPU, $THREADS threads each) on $(hostname)"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ibmarl
cd "$REPO"

python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA on $(hostname)"; exit 1; }
[ -f "config/legacy/$SCENARIO/$(python -c "from analysis.paper_run import VARIANTS; print(VARIANTS['$V'][0])").yaml" ] \
  || { echo "no frozen base for $SCENARIO/$V under config/legacy -- git pull?"; exit 1; }

set +e
timeout --signal=INT "$RUN_TIMEOUT" python -m analysis.paper_sweep \
  --scenario "$SCENARIO" --tag "$TAG" --only "$V" --seeds "$S" \
  --base-config "$BASE_CONFIG" \
  --jobs "$JOBS" --threads "$THREADS" --resume-interval "$RESUME_INTERVAL"
rc=$?
set -e
if [ $rc -eq 124 ]; then
  echo "walltime: stopped at $RUN_TIMEOUT, resubmitting index $SLURM_ARRAY_TASK_ID to continue from the group's resume points"
  sbatch -J "$SLURM_JOB_NAME" --export=ALL --array="$SLURM_ARRAY_TASK_ID" "$SCRIPT"
  exit 0
fi
exit $rc
