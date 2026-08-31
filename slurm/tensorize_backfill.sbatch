#!/bin/bash
#SBATCH --job-name=g2c-tenfill
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
# Tensorise the backfilled episodes into a staging directory.  They are kept out
# of `tensor/` so the running headline job's episode split stays reproducible;
# merging them is a `mv` once that run is finished and evaluated.
set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
DATA=/lustre/fs1/home/mu290328/datasets/GTraffic2Crash
mkdir -p "$DATA/raw_backfill"
"$REPO/.venv/bin/python" - <<'PY'
import os, shutil
raw="/lustre/fs1/home/mu290328/datasets/GTraffic2Crash/raw"
ten="/lustre/fs1/home/mu290328/datasets/GTraffic2Crash/tensor"
stage="/lustre/fs1/home/mu290328/datasets/GTraffic2Crash/raw_backfill"
done_=set(os.listdir(ten))
n=0
for f in sorted(os.listdir(raw)):
    if f.endswith(".npz") and f not in done_ and not os.path.exists(os.path.join(stage,f)):
        os.symlink(os.path.join(raw,f), os.path.join(stage,f)); n+=1
print(f"staged {n} episodes for tensorisation")
PY
"$REPO/.venv/bin/python" "$REPO/scripts/tensorize_episodes.py" \
    --raw "$DATA/raw_backfill" --out "$DATA/tensor_backfill" --stride 5 --workers 32
echo "backfill tensorised; merge with: mv $DATA/tensor_backfill/*.npz $DATA/tensor/"
