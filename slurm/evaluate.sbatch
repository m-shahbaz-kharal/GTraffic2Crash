#!/bin/bash
#SBATCH --job-name=g2c-eval
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
set -u
cd /lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
RUN="${RUN:-pilot_small}"
CK="${CK:-$(ls -t runs/$RUN/checkpoints/best-*.ckpt 2>/dev/null | head -1)}"
echo "evaluating $CK"
.venv/bin/python scripts/evaluate.py --checkpoint "$CK" \
    --out "runs/$RUN/evaluation" --max-batches "${MAXB:-120}" --videos 3
