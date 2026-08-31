#!/bin/bash
#SBATCH --job-name=g2c-resume
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=12
#SBATCH --mem=140G
#SBATCH --time=00:40:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
# The headline run is 39 h against a 36 h limit, so it will be requeued and must
# resume.  This proves that path on a throwaway run rather than discovering it
# fails a day and a half in.
set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
cd "$REPO"
export WANDB_MODE=offline WANDB_DIR="$REPO/runs" OMP_NUM_THREADS=6
OUT="$REPO/runs/resume_test"
rm -rf "$OUT"

echo "======== FIRST LEG: train to step 60 ========"
.venv/bin/python scripts/train.py --run resume_test --size small --devices 1 \
    --batch-size 4 --max-steps 60 --warmup-steps 5 --workers 6 \
    --train-stride 60 --val-stride 60 --limit-val-batches 4 \
    --val-check-interval 1.0 --epochs 1 2>&1 | grep -E "^\[step|\[train\]|epoch "

echo; echo "======== SECOND LEG: resume and continue to 120 ========"
CK="$OUT/checkpoints/last.ckpt"
[ -f "$CK" ] || { echo "FAIL: no last.ckpt written"; exit 1; }
.venv/bin/python scripts/train.py --run resume_test --size small --devices 1 \
    --batch-size 4 --max-steps 120 --warmup-steps 5 --workers 6 \
    --train-stride 60 --val-stride 60 --limit-val-batches 4 \
    --val-check-interval 1.0 --epochs 2 --resume "$CK" 2>&1 | grep -E "^\[step|\[train\]|Restor|epoch "

echo; echo "======== RESULT ========"
FIRST=$(grep -c "^\[step" "$REPO/logs/g2c-resume-${SLURM_JOB_ID}.out" 2>/dev/null || echo 0)
echo "if the second leg's first heartbeat is at a step above 60, resume works"
