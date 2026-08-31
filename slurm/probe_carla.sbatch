#!/bin/bash
#SBATCH --job-name=g2c-probe
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%j.out

# Measure CARLA Town05 throughput and collision behaviour before committing to a
# data-generation plan.  Runs several configurations against one server so the
# comparison is apples to apples.

set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
CARLA_ROOT="$HOME/software/CARLA_0.9.15"
VENV_PY="$HOME/software/carla-venv/bin/python"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
OUT="$REPO/logs/probe-${SLURM_JOB_ID}"
mkdir -p "$OUT"

RPC_PORT=$(( 2000 + (SLURM_JOB_ID % 400) * 5 ))
echo "node=$(hostname)  job=${SLURM_JOB_ID}  rpc=${RPC_PORT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source "$HOME/software/carla-scripts/vulkan_setup.sh"

SERVER_LOG="$REPO/logs/carla-server-${SLURM_JOB_ID}.log"
cd "$CARLA_ROOT" || exit 1
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic \
    -carla-rpc-port=${RPC_PORT} -graphicsadapter=0 > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill ${SERVER_PID} 2>/dev/null; sleep 5; kill -9 ${SERVER_PID} 2>/dev/null; }
trap cleanup EXIT

echo "waiting for server on ${RPC_PORT} ..."
READY=0
for i in $(seq 1 150); do
    kill -0 ${SERVER_PID} 2>/dev/null || { echo "server died at ${i}"; break; }
    if "$VENV_PY" -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(('127.0.0.1',${RPC_PORT}))==0 else 1)" 2>/dev/null; then
        echo "server up after ~$((i*5))s"; READY=1; break
    fi
    sleep 5
done
[ "$READY" -ne 1 ] && { tail -40 "$SERVER_LOG"; exit 1; }

run () {
    echo; echo "======== $1 ========"
    shift
    "$VENV_PY" "$REPO/scripts/probe_carla.py" --port ${RPC_PORT} "$@" 2>&1 | tail -50
}

# 1. Baseline: does rendering cost us anything when we never read a camera?
run "A  200 vehicles, rendering on"    --vehicles 200 --seconds 40 --out "$OUT/a.json"
run "B  200 vehicles, no rendering"    --vehicles 200 --seconds 40 --no-render --out "$OUT/b.json"
# 2. Scale: how far can the vehicle count go before ticks/s falls over?
run "C  350 vehicles, no rendering"    --vehicles 350 --seconds 40 --no-render --out "$OUT/c.json"
# 3. The question that decides the whole bad-actor design: do they actually crash?
run "D  250 vehicles, 15% aggressive"  --vehicles 250 --seconds 120 --no-render \
    --aggressive-frac 0.15 --collision-sensors --out "$OUT/d.json"
run "E  250 vehicles, 40% aggressive"  --vehicles 250 --seconds 120 --no-render \
    --aggressive-frac 0.40 --collision-sensors --out "$OUT/e.json"
# 4. Does the tick rate survive a finer timestep (needed for clean collisions)?
run "F  250 vehicles, dt=0.025"        --vehicles 250 --seconds 40 --no-render --delta 0.025 \
    --out "$OUT/f.json"

echo; echo "======== SUMMARY ========"
"$VENV_PY" - "$OUT" <<'PY'
import json, glob, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f))
    rows.append((os.path.basename(f)[0].upper(), d.get("n_spawned"), d.get("args",{}).get("no_render"),
                 d.get("args",{}).get("delta"), d.get("args",{}).get("aggressive_frac"),
                 d.get("ticks_per_wall_second"), d.get("sim_seconds_per_wall_second"),
                 d.get("state_readout_fraction"), d.get("collisions_per_sim_minute"),
                 (d.get("speed_m_s") or {}).get("p50"), (d.get("speed_m_s") or {}).get("p95")))
print(f"{'run':>3} {'veh':>4} {'norend':>6} {'dt':>6} {'aggr':>5} {'tick/s':>7} {'xreal':>6} "
      f"{'readout':>7} {'crash/min':>9} {'v50':>5} {'v95':>5}")
for r in rows:
    print(f"{r[0]:>3} {str(r[1]):>4} {str(r[2]):>6} {str(r[3]):>6} {str(r[4]):>5} "
          f"{str(r[5]):>7} {str(r[6]):>6} {str(r[7]):>7} {str(r[8]):>9} {str(r[9]):>5} {str(r[10]):>5}")
PY
