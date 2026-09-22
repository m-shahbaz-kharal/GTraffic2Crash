#!/bin/bash
#SBATCH --job-name=g2c-train
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out

# Resumable by construction: Lightning writes `last.ckpt` at every validation and
# this script picks it up automatically.
#
# Note what --requeue does and does not cover.  It requeues after preemption or
# node failure.  It does NOT requeue a job SLURM cancels for exhausting its time
# limit -- that is terminal, and the first headline run died that way at step
# 27,700 of 32,000 after exactly 36 hours, with nothing to restart it.  The time
# limit is now set generously and a run that stops short simply needs this script
# resubmitting; it will pick up last.ckpt and continue.
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

# SLURM allocating two GPUs does not guarantee the node hands over two: job
# 833094 was allocated gres/gpu=2 on evc30, saw one, and died two minutes in
# because --devices was hardcoded at 2.  Take what is actually visible and
# raise accumulation to keep the effective batch, and therefore the schedule,
# exactly where it was.
GPUS=$(nvidia-smi -L | wc -l)
if [ "$GPUS" -lt 1 ]; then
    echo "no GPU visible; refusing to start" >&2
    exit 1
fi
WANT="${WANT_GPUS:-2}"
if [ "$GPUS" -lt "$WANT" ]; then
    ACCUM=$(( ACCUM * WANT / GPUS ))
    echo "only $GPUS of $WANT GPUs visible: accumulate raised to $ACCUM"
fi

echo "run=$RUN size=$SIZE gpus=$GPUS batch=$BATCH accum=$ACCUM epochs=$EPOCHS lr=$LR"
echo "effective batch = $(( BATCH * ACCUM * GPUS ))"

RESUME=""
if [ -f "$OUT/checkpoints/last.ckpt" ]; then
    RESUME="--resume $OUT/checkpoints/last.ckpt"
    echo "resuming from $OUT/checkpoints/last.ckpt"
fi

# No srun: see the note in pilot.sbatch -- it pins the task to one CPU.
"$PY" scripts/train.py \
    --run "$RUN" --size "$SIZE" --devices "$GPUS" \
    --batch-size "$BATCH" --accumulate "$ACCUM" --epochs "$EPOCHS" \
    --learning-rate "$LR" --workers 12 $RESUME $EXTRA
