#!/bin/bash
# Weights & Biases runs offline on this cluster.  After `wandb login`, this
# uploads every finished run.
set -e
PY=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/.venv/bin
"$PY/wandb" login
for d in /lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/runs/wandb/offline-run-*; do
    echo "syncing $d"; "$PY/wandb" sync "$d"
done
