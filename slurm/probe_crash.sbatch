#!/bin/bash
#SBATCH --job-name=g2c-crash
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
OUT="$REPO/logs/crash-${SLURM_JOB_ID}"; mkdir -p "$OUT"
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

for FRAC in 0.00 0.02 0.05 0.10; do
  echo; echo "======== bad-actor fraction ${FRAC} ========"
  "$VENV_PY" "$REPO/scripts/probe_crash_rate.py" --port ${RPC_PORT} --vehicles 160 \
      --seconds 180 --warmup 60 --bad-actor-fraction ${FRAC} --out "$OUT/f${FRAC}.json" 2>&1 | tail -40
done

echo; echo "======== CRASH CALIBRATION SUMMARY ========"
"$VENV_PY" - "$OUT" <<'PY'
import json, glob, os, sys
print(f"{'bad%':>5} {'n_bad':>5} {'all/min':>8} {'>1m/s':>7} {'>3m/s':>7} {'>5m/s':>7} {'>8m/s':>7} "
      f"{'imp_p50':>8} {'imp_p90':>8} {'spd_p50':>8} {'spd_p90':>8}")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f))
    print(f"{d['bad_actor_fraction']*100:>5.0f} {d['n_bad']:>5} {d['collisions_per_minute']:>8} "
          f"{d['per_minute_impact_over_1ms']:>7} {d['per_minute_impact_over_3ms']:>7} "
          f"{d['per_minute_impact_over_5ms']:>7} {d['per_minute_impact_over_8ms']:>7} "
          f"{d['impulse']['p50']:>8.0f} {d['impulse']['p90']:>8.0f} "
          f"{d['impact_speed_m_s']['p50']:>8.2f} {d['impact_speed_m_s']['p90']:>8.2f}")
    if d.get('pairs_over_3ms'):
        print("      who crashes:", d['pairs_over_3ms'])
PY
