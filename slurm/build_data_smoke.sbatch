#!/bin/bash
#SBATCH --job-name=g2c-smoke
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out

# Build the static map, then generate two short episodes -- one nominal, one with
# bad actors -- so the whole generation path can be checked before committing to
# hundreds of episodes.
set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
DATA=/lustre/fs1/home/mu290328/datasets/GTraffic2Crash
CARLA_ROOT="$HOME/software/CARLA_0.9.15"
VENV_PY="$HOME/software/carla-venv/bin/python"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
mkdir -p "$DATA/raw" "$DATA/meta"
RPC_PORT=$(( 2000 + (SLURM_JOB_ID % 400) * 5 ))
echo "node=$(hostname) job=${SLURM_JOB_ID} rpc=${RPC_PORT}"
source "$HOME/software/carla-scripts/vulkan_setup.sh"
cd "$CARLA_ROOT" || exit 1
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low \
    -carla-rpc-port=${RPC_PORT} -graphicsadapter=0 \
    > "$REPO/logs/carla-server-${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null; sleep 5; kill -9 ${SERVER_PID} 2>/dev/null' EXIT
for i in $(seq 1 150); do
    "$VENV_PY" -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(('127.0.0.1',${RPC_PORT}))==0 else 1)" 2>/dev/null && { echo "server up ~$((i*5))s"; break; }
    sleep 5
done

echo "======== STATIC MAP ========"
"$VENV_PY" "$REPO/scripts/build_static_map.py" --port ${RPC_PORT} \
    --out "$DATA/meta/town05_static.npz" || exit 1

echo "======== EPISODE 9000 (nominal) ========"
"$VENV_PY" "$REPO/scripts/generate_episode.py" --port ${RPC_PORT} --episode 9000 \
    --vehicles 160 --seconds 120 --warmup 60 --bad-actor-fraction 0.0 \
    --out "$DATA/raw/ep_09000.npz" || exit 1

echo "======== EPISODE 9001 (bad actors) ========"
"$VENV_PY" "$REPO/scripts/generate_episode.py" --port ${RPC_PORT} --episode 9001 \
    --vehicles 160 --seconds 120 --warmup 60 --bad-actor-fraction 0.05 \
    --out "$DATA/raw/ep_09001.npz" || exit 1

ls -la "$DATA/raw" "$DATA/meta"
