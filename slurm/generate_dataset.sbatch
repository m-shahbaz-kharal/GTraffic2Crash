#!/bin/bash
#SBATCH --job-name=g2c-gen
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --exclude=evc46,evc47,evc48,evc49,evc50
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --output=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%A_%a.out
#SBATCH --error=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash/logs/%x-%A_%a.out

# One array task runs two CARLA servers side by side, each generating a slice of
# the episode list.  Episodes already on disk are skipped, so a failed task can
# simply be resubmitted.
set -u
REPO=/lustre/fs1/home/mu290328/github-repos/GTraffic2Crash
DATA=/lustre/fs1/home/mu290328/datasets/GTraffic2Crash
CARLA_ROOT="$HOME/software/CARLA_0.9.15"
VENV_PY="$HOME/software/carla-venv/bin/python"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
mkdir -p "$DATA/raw"

N_TASKS=20
SERVERS_PER_TASK=2
TOTAL_EPISODES=400
source "$HOME/software/carla-scripts/vulkan_setup.sh"

start_server () {   # $1 = rpc port, $2 = log
    cd "$CARLA_ROOT" || exit 1
    ./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low \
        -carla-rpc-port=$1 -graphicsadapter=0 > "$2" 2>&1 &
    echo $!
}
wait_server () {
    for i in $(seq 1 180); do
        "$VENV_PY" -c "
import socket,sys
s=socket.socket(); s.settimeout(2)
sys.exit(0 if s.connect_ex(('127.0.0.1',$1))==0 else 1)" 2>/dev/null && return 0
        sleep 5
    done
    return 1
}

PIDS=()
for k in $(seq 0 $((SERVERS_PER_TASK-1))); do
    PORT=$(( 3000 + SLURM_ARRAY_TASK_ID * 20 + k * 6 ))
    LOG="$REPO/logs/carla-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${k}.log"
    PID=$(start_server $PORT "$LOG")
    PIDS+=($PID)
    echo "server $k pid $PID port $PORT"
done
cleanup () { for p in "${PIDS[@]}"; do kill $p 2>/dev/null; done; sleep 5
             for p in "${PIDS[@]}"; do kill -9 $p 2>/dev/null; done; }
trap cleanup EXIT

run_slice () {   # $1 = server index
    local k=$1
    local PORT=$(( 3000 + SLURM_ARRAY_TASK_ID * 20 + k * 6 ))
    wait_server $PORT || { echo "server $k failed to start"; return 1; }
    echo "server $k up on $PORT"
    local WORKER=$(( SLURM_ARRAY_TASK_ID * SERVERS_PER_TASK + k ))
    local N_WORKERS=$(( N_TASKS * SERVERS_PER_TASK ))
    for (( ep=WORKER; ep<TOTAL_EPISODES; ep+=N_WORKERS )); do
        OUT=$(printf "%s/raw/ep_%05d.npz" "$DATA" $ep)
        if [ -f "$OUT" ]; then echo "episode $ep already done"; continue; fi
        CFG=$("$VENV_PY" "$REPO/scripts/episode_config.py" $ep)
        VEH=$(echo "$CFG"  | "$VENV_PY" -c "import json,sys;print(json.load(sys.stdin)['vehicles'])")
        FRAC=$(echo "$CFG" | "$VENV_PY" -c "import json,sys;print(json.load(sys.stdin)['bad_actor_fraction'])")
        SEC=$(echo "$CFG"  | "$VENV_PY" -c "import json,sys;print(json.load(sys.stdin)['seconds'])")
        WARM=$(echo "$CFG" | "$VENV_PY" -c "import json,sys;print(json.load(sys.stdin)['warmup'])")
        "$VENV_PY" "$REPO/scripts/generate_episode.py" --port $PORT --episode $ep \
            --vehicles $VEH --seconds $SEC --warmup $WARM \
            --bad-actor-fraction $FRAC --out "$OUT" 2>&1 | sed "s/^/[s$k] /"
    done
}

for k in $(seq 0 $((SERVERS_PER_TASK-1))); do run_slice $k & done
wait
echo "task ${SLURM_ARRAY_TASK_ID} finished; $(ls "$DATA/raw" | wc -l) episodes on disk"
