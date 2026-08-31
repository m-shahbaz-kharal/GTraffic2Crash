#!/bin/bash
#SBATCH --job-name=g2c-ddp
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out

# A few minutes of two-GPU training, purely to prove the distributed path works
# before the headline run commits a day to it.  The known hazard is the last
# decoder block's global-vector parameters, which receive no gradient and would
# otherwise make DDP raise; the strategy is set to tolerate them.
set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
cd "$REPO"
export WANDB_MODE=offline WANDB_DIR="$REPO/runs" OMP_NUM_THREADS=6
# Newer SLURM binds a plain `srun` inside a batch job to one CPU, which
# silently throttles the data loader to a single worker.
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# No srun: inside a batch job it binds the task to a single CPU, which starves
# the data loader (AveCPU showed 1.1 cores of a 14-core allocation).  Lightning
# spawns its own worker processes for multi-GPU on one node, so srun buys nothing.
.venv/bin/python scripts/train.py --run ddp_smoke --size small --devices 2 \
    --batch-size 4 --epochs 1 --max-steps 60 --warmup-steps 10 --workers 4 \
    --limit-val-batches 8 --train-stride 40 --val-stride 40
