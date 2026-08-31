#!/bin/bash
#SBATCH --job-name=g2c-overfit
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
set -u
cd /lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
echo "======== small ========"
.venv/bin/python scripts/overfit_check.py --size small --steps 400 --batch-size 4
echo; echo "======== base ========"
.venv/bin/python scripts/overfit_check.py --size base --steps 250 --batch-size 4
