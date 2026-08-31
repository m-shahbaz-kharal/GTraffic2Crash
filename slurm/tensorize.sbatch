#!/bin/bash
#SBATCH --job-name=g2c-tensorize
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
DATA=/lustre/fs1/home/mu290328/datasets/GTraffic2Crash
"$REPO/.venv/bin/python" "$REPO/scripts/tensorize_episodes.py" \
    --raw "$DATA/raw" --out "$DATA/tensor" --stride 5 --workers 32
echo "======== DATASET SUMMARY ========"
cd "$REPO" && "$REPO/.venv/bin/python" scripts/visualise_dataset.py --n-videos 3
