#!/bin/bash
#SBATCH --job-name=g2c-pilot
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=14
#SBATCH --mem=140G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out

# Pilot: the small model on two GPUs.  Two jobs at once -- it exercises the
# distributed path that the headline run will use, and it produces real learning
# curves within the hour instead of at the end of a day.
set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
cd "$REPO"
export WANDB_MODE=offline WANDB_DIR="$REPO/runs" OMP_NUM_THREADS=6
# Newer SLURM binds a plain `srun` inside a batch job to one CPU, which
# silently throttles the data loader to a single worker.
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
nvidia-smi --query-gpu=index,name --format=csv,noheader
OUT="$REPO/runs/pilot_small"
RESUME=""
[ -f "$OUT/checkpoints/last.ckpt" ] && RESUME="--resume $OUT/checkpoints/last.ckpt"
# No srun: inside a batch job it binds the task to a single CPU, which starves
# the data loader (AveCPU showed 1.1 cores of a 14-core allocation).  Lightning
# spawns its own worker processes for multi-GPU on one node, so srun buys nothing.
.venv/bin/python scripts/train.py --run pilot_small --size small --devices 1 \
    --batch-size 8 --epochs 6 --learning-rate 6e-4 --warmup-steps 400 \
    --workers 8 --train-stride 2 --val-stride 5 --val-check-interval 0.08 --limit-val-batches 120 $RESUME
