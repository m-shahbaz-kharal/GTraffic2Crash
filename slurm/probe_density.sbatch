#!/bin/bash
#SBATCH --job-name=g2c-density
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out

# How many vehicles can Town05 hold before it gridlocks?  The first probe
# measured a median speed of ~0 at 200 vehicles, which would make forecasting
# degenerate towards "copy the last frame".  This sweeps the fleet size with a
# proper 90 s warm-up so the answer is not an artefact of vehicles spawning at
# rest.
set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
CARLA_ROOT="$HOME/software/CARLA_0.9.15"
VENV_PY="$HOME/software/carla-venv/bin/python"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
OUT="$REPO/logs/density-${SLURM_JOB_ID}"; mkdir -p "$OUT"
RPC_PORT=$(( 2000 + (SLURM_JOB_ID % 400) * 5 ))
echo "node=$(hostname) job=${SLURM_JOB_ID} rpc=${RPC_PORT}"
source "$HOME/software/carla-scripts/vulkan_setup.sh"
SERVER_LOG="$REPO/logs/carla-server-${SLURM_JOB_ID}.log"
cd "$CARLA_ROOT" || exit 1
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low \
    -carla-rpc-port=${RPC_PORT} -graphicsadapter=0 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null; sleep 5; kill -9 ${SERVER_PID} 2>/dev/null' EXIT
for i in $(seq 1 150); do
    "$VENV_PY" -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(('127.0.0.1',${RPC_PORT}))==0 else 1)" 2>/dev/null && { echo "server up ~$((i*5))s"; break; }
    sleep 5
done

for N in 60 100 140 200 260; do
  echo; echo "======== $N vehicles ========"
  "$VENV_PY" "$REPO/scripts/probe_carla.py" --port ${RPC_PORT} --vehicles $N \
      --seconds 60 --warmup 90 --no-render --out "$OUT/n${N}.json" 2>&1 | tail -45
done

echo; echo "======== DENSITY SUMMARY ========"
"$VENV_PY" - "$OUT" <<'PY'
import json, glob, os, sys
print(f"{'veh':>4} {'tick/s':>7} {'xreal':>6} {'moving':>7} {'>5m/s':>6} {'mean':>6} {'p50':>6} {'p95':>6}")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json")), key=lambda x: int(''.join(c for c in os.path.basename(x) if c.isdigit()))):
    d = json.load(open(f)); s = d.get("speed_m_s", {})
    print(f"{d['n_spawned']:>4} {d['ticks_per_wall_second']:>7} {d['sim_seconds_per_wall_second']:>6} "
          f"{d.get('moving_fraction'):>7} {d.get('fraction_over_5ms'):>6} {s.get('mean'):>6} {s.get('p50'):>6} {s.get('p95'):>6}")
    print("     warm-up:", " ".join(f"{t['sim_time_s']:.0f}s:{t['moving_frac']:.2f}" for t in d.get("warmup_trace", [])))
PY
