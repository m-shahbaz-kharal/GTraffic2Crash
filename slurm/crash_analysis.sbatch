#!/bin/bash
#SBATCH --job-name=g2c-crash
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
set -u
cd /lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
RUN="${RUN:-pilot_small}"
CK="${CK:-$(ls -t runs/$RUN/checkpoints/best-*.ckpt | head -1)}"
echo "analysing $CK"
.venv/bin/python scripts/crash_anticipation.py --checkpoint "$CK" \
    --out "runs/$RUN/evaluation" --max-windows "${MAXW:-1200}"
