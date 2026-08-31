#!/bin/bash
#SBATCH --job-name=g2c-train
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=36:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out

# Resumable by construction: Lightning writes `last.ckpt` every epoch and this
# script picks it up automatically, so a requeue after preemption costs at most
# one epoch rather than the whole run.
set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
PY="$REPO/.venv/bin/python"
RUN="${RUN:-base_v1}"
SIZE="${SIZE:-base}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-2}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
EXTRA="${EXTRA:-}"
OUT="$REPO/runs/$RUN"
mkdir -p "$OUT"
cd "$REPO"

export WANDB_MODE=offline
export WANDB_DIR="$REPO/runs"
export OMP_NUM_THREADS=8
# Newer SLURM binds a plain `srun` inside a batch job to one CPU, which
# silently throttles the data loader to a single worker.
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "run=$RUN size=$SIZE batch=$BATCH accum=$ACCUM epochs=$EPOCHS lr=$LR"

RESUME=""
if [ -f "$OUT/checkpoints/last.ckpt" ]; then
    RESUME="--resume $OUT/checkpoints/last.ckpt"
    echo "resuming from $OUT/checkpoints/last.ckpt"
fi

# No srun: see the note in pilot.sbatch -- it pins the task to one CPU.
"$PY" scripts/train.py \
    --run "$RUN" --size "$SIZE" --devices 2 \
    --batch-size "$BATCH" --accumulate "$ACCUM" --epochs "$EPOCHS" \
    --learning-rate "$LR" --workers 12 $RESUME $EXTRA
