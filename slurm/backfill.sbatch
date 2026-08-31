#!/bin/bash
#SBATCH --job-name=g2c-fill
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%A_%a.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%A_%a.out

# Regenerate the episodes the first pass lost to Traffic Manager port collisions.
set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
DATA=/lustre/fs1/home/mu290328/datasets/GTraffic2Crash
CARLA_ROOT="$HOME/software/CARLA_0.9.15"
VENV_PY="$HOME/software/carla-venv/bin/python"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
source "$HOME/software/carla-scripts/vulkan_setup.sh"

MISSING=$(cat "$REPO/slurm/missing_episodes.txt")
PORT=$(( 5000 + SLURM_ARRAY_TASK_ID * 10 ))
cd "$CARLA_ROOT" || exit 1
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low \
    -carla-rpc-port=${PORT} -graphicsadapter=0 \
    > "$REPO/logs/carla-fill-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null' EXIT
for i in $(seq 1 180); do
    "$VENV_PY" -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(('127.0.0.1',${PORT}))==0 else 1)" 2>/dev/null && break
    sleep 5
done
echo "server up on ${PORT}"

k=0
for ep in $MISSING; do
    if [ $(( k % 12 )) -eq "$SLURM_ARRAY_TASK_ID" ]; then
        OUT=$(printf "%s/raw/ep_%05d.npz" "$DATA" $ep)
        if [ ! -f "$OUT" ]; then
            CFG=$("$VENV_PY" "$REPO/scripts/episode_config.py" $ep)
            VEH=$(echo "$CFG"  | "$VENV_PY" -c "import json,sys;print(json.load(sys.stdin)['vehicles'])")
            FRAC=$(echo "$CFG" | "$VENV_PY" -c "import json,sys;print(json.load(sys.stdin)['bad_actor_fraction'])")
            "$VENV_PY" "$REPO/scripts/generate_episode.py" --port $PORT --episode $ep \
                --vehicles $VEH --seconds 450 --warmup 60 \
                --bad-actor-fraction $FRAC --out "$OUT" 2>&1 | sed "s/^/[fill] /"
        fi
    fi
    k=$((k+1))
done
echo "backfill task ${SLURM_ARRAY_TASK_ID} done"
