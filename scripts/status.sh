#!/bin/bash
# One-line-per-thing view of everything in flight.
cd /lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
DATA=/lustre/fs1/home/mu290328/datasets/GTraffic2Crash
echo "=== $(date +%H:%M) ==="
echo "episodes generated : $(ls $DATA/raw 2>/dev/null | wc -l)/400"
echo "episodes tensorised: $(ls $DATA/tensor 2>/dev/null | wc -l)"
echo "jobs:"
squeue -u "$USER" -r -h -o "  %.10i %.12j %.9T %.7M %R" | grep -v g2c-gen
GEN=$(squeue -u "$USER" -r -h -n g2c-gen | wc -l)
[ "$GEN" -gt 0 ] && echo "  ($GEN generation tasks running)"
for run in pilot_small base_v1; do
  L=$(ls -t logs/*"${run%%_*}"*.out 2>/dev/null | head -1)
  if [ -f "runs/$run/reports/validation.jsonl" ]; then
    echo "--- $run ---"
    .venv/bin/python scripts/report_run.py "$run" 2>/dev/null | sed -n '1,3p;/epoch/,/^$/p' | head -12
  fi
done
for f in logs/g2c-pilot-*.out logs/g2c-train-*.out; do
  [ -f "$f" ] || continue
  S=$(grep "^\[step" "$f" 2>/dev/null | tail -1)
  [ -n "$S" ] && echo "$(basename $f): $S"
done
