#!/bin/bash
#SBATCH --job-name=g2c-smoketrain
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
PY="$REPO/.venv/bin/python"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "======== THROUGHPUT ========"
"$PY" scripts/benchmark_model.py

echo; echo "======== PLUMBING: can a small model memorise 2 batches? ========"
# If a network cannot drive the loss down on a handful of windows it has seen
# hundreds of times, the fault is in the plumbing, not the data.
"$PY" scripts/train.py --run smoke_overfit --size small --epochs 40 \
    --overfit-batches 2 --batch-size 4 --workers 2 --warmup-steps 20 \
    --learning-rate 1e-3 --no-augment --val-check-interval 1.0 2>&1 | tail -60
