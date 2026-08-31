#!/bin/bash
# Tensorise everything generated so far, then launch the headline run behind it.
# Both steps are idempotent: tensorisation skips episodes already done, and
# training resumes from `last.ckpt` if the job is requeued.
set -eu
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
cd "$REPO"
TJOB=$(sbatch --parsable slurm/tensorize.sbatch)
echo "tensorise: $TJOB"
RUN=${RUN:-base_v1} SIZE=${SIZE:-base} BATCH=${BATCH:-8} ACCUM=${ACCUM:-1} \
EPOCHS=${EPOCHS:-4} LR=${LR:-2e-4} \
EXTRA="${EXTRA:---warmup-steps 1000 --max-steps 32000 --train-stride 1 --val-stride 5}" \
  sbatch --dependency=afterok:$TJOB slurm/train.sbatch
