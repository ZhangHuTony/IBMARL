#!/bin/bash
# One (variant, seed) run per array task -- 8 variants x 5 seeds = 40 tasks.
#
#   sbatch slurm/run_sweep.slurm                 # the whole sweep
#   sbatch --array=5,12 slurm/run_sweep.slurm    # re-run two killed tasks
#
# One run per task because the 8 h cap does not fit 5 seeds of one variant at
# --jobs 3 on a 2080 (~7-8 h per IBMARL run when three share a GPU).  A run
# alone on its GPU is well inside 8 h, and paper_run's resume point every
# `--resume-interval` iterations means a walltime kill costs at most that many
# iterations: resubmit the same index and it continues.  Results land in the
# canonical results/<tag>/<variant>/seed_N layout that analysis/paper3_figures.py
# reads directly (buzzwire4 needed symlinks because each task had its own tag).
#SBATCH --job-name=bw5
#SBATCH --account=dbrown-gpu-np
#SBATCH --partition=dbrown-gpu-np
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-39
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1183156/logs/bw5-%A_%a.out

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ibmarl
cd ~/code/IBMARL

python -c "import torch; assert torch.cuda.is_available()" || { echo "no CUDA on $(hostname)"; exit 1; }

TAG=buzzwire5
SEEDS=5
METHODS=(bc_eval maddpg rlfd rft ibmarl ibmarl_gated_a0.4 ibmarl_strict_gated_a0.4 ibmarl_gated_a0.4_1critic)
V=${METHODS[$((SLURM_ARRAY_TASK_ID / SEEDS))]}
S=$((SLURM_ARRAY_TASK_ID % SEEDS))
echo "task $SLURM_ARRAY_TASK_ID -> $V seed $S on $(hostname)  (tag $TAG)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -m analysis.paper_run \
  --variant "$V" --seed "$S" --scenario buzz_wire --tag "$TAG" \
  --resume-interval 5
