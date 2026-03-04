#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_ID=0
CONFIG_FILE="config.ini"
PARAM_FOLDER=2000

LATENCIES=(0.200 0.225 0.250 0.275 0.300 0.325 0.350 0.375 0.400)

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "Running in: $(pwd)"
echo "Using python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "Config file: $CONFIG_FILE"
echo "PARAM_FOLDER: $PARAM_FOLDER"
echo

set_ini_value () {
  local section="$1"
  local key="$2"
  local value="$3"
  local tmp
  tmp="$(mktemp)"

  awk -v section="$section" -v key="$key" -v value="$value" '
    BEGIN { in_section=0; changed=0 }
    /^\[/ { in_section = ($0 == "[" section "]") }
    in_section && $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
      sub(/=.*/, "= " value)
      changed=1
    }
    { print }
    END { if (changed == 0) exit 2 }
  ' "$CONFIG_FILE" > "$tmp"

  mv "$tmp" "$CONFIG_FILE"
}

make_ddqn_dirs () {
  mkdir -p logs/rl/ddqn/{epsilon,loss,models,reward,splits,system}
}

mkdir -p logs logs/rl

for lat in "${LATENCIES[@]}"; do
  safe_lat="${lat/./p}"   # 0.200 -> 0p200 (safe folder name)

  echo "=============================="
  echo "[$(ts)] DDQN run: MAX_INFERENCE_LATENCY=$lat (folder=$PARAM_FOLDER)"
  echo "=============================="

  # clean ddqn folder for this run
  rm -rf logs/rl/ddqn
  make_ddqn_dirs

  # ---- set DDQN configuration ----
  set_ini_value "ALGORITHM" "SPLIT_ALGORITHM" "2"           # RL-based split
  set_ini_value "ALGORITHM" "RL_ALGORITHM" "1"              # 1=DDQN
  set_ini_value "ALGORITHM" "INFERENCE" "0"
  set_ini_value "ALGORITHM" "N_EPISODES" "5000"
  set_ini_value "ALGORITHM" "START_EPISODE" "1"
  set_ini_value "ALGORITHM" "PARAM_PATH" "$PARAM_FOLDER"
  set_ini_value "ALGORITHM" "MAX_INFERENCE_LATENCY" "$lat"

  # DDQN hyperparams
  set_ini_value "DRL_HYPERPARAMETERS" "LR" "0.00001"        # 1e-5
  set_ini_value "DRL_HYPERPARAMETERS" "TAU" "0.001"         # 1e-3

  echo "[$(ts)] Config check:"
  echo "  $(grep -m1 '^SPLIT_ALGORITHM' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^RL_ALGORITHM' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^N_EPISODES' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PARAM_PATH' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^MAX_INFERENCE_LATENCY' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^LR' "$CONFIG_FILE" | head -1)"
  echo "  $(grep -m1 '^TAU' "$CONFIG_FILE")"
  echo

  # save config + stdout log
  cp "$CONFIG_FILE" "logs/config_ddqn_lat_${safe_lat}.ini"
  run_log="logs/output_ddqn_lat_${safe_lat}.log"

  echo "[$(ts)] Starting python (log: $run_log)"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py 2>&1 | tee "$run_log"

  # archive ddqn logs for this latency
  dest="logs/rl/ddqn_lat_${safe_lat}"
  if [[ -e "$dest" ]]; then
    echo "ERROR: destination already exists: $dest"
    exit 1
  fi
  mv logs/rl/ddqn "$dest"

  echo "[$(ts)] Finished DDQN run (lat=$lat). Saved to $dest"
  echo
done

echo "[$(ts)] All DDQN latency runs completed."