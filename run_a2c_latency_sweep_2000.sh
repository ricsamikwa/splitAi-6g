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

make_a2c_dirs () {
  mkdir -p logs/rl/a2c/{epsilon,loss,models,reward,splits,system}
}

mkdir -p logs logs/rl

for lat in "${LATENCIES[@]}"; do
  safe_lat="${lat/./p}"

  echo "=============================="
  echo "[$(ts)] A2C run: MAX_INFERENCE_LATENCY=$lat"
  echo "=============================="

  rm -rf logs/rl/a2c
  make_a2c_dirs

  # A2C configuration
  set_ini_value "ALGORITHM" "SPLIT_ALGORITHM" "2"
  set_ini_value "ALGORITHM" "RL_ALGORITHM" "2"
  set_ini_value "ALGORITHM" "N_EPISODES" "5000"
  set_ini_value "ALGORITHM" "ACCURACY_DECREASE" "80"
  set_ini_value "ALGORITHM" "PARAM_PATH" "$PARAM_FOLDER"
  set_ini_value "ALGORITHM" "MAX_INFERENCE_LATENCY" "$lat"

  # Same A2C learning params as before
  set_ini_value "DRL_HYPERPARAMETERS" "LR_ACTOR" "0.0001"
  set_ini_value "DRL_HYPERPARAMETERS" "LR_CRITIC" "0.0005"
  set_ini_value "DRL_HYPERPARAMETERS" "ENTROPY_FACTOR" "0.005"

  echo "[$(ts)] Config check:"
  echo "  $(grep -m1 '^SPLIT_ALGORITHM' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^RL_ALGORITHM' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^N_EPISODES' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^ACCURACY_DECREASE' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PARAM_PATH' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^MAX_INFERENCE_LATENCY' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^LR_ACTOR' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^LR_CRITIC' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^ENTROPY_FACTOR' "$CONFIG_FILE")"
  echo

  cp "$CONFIG_FILE" "logs/config_a2c_lat_${safe_lat}.ini"
  run_log="logs/output_a2c_lat_${safe_lat}.log"

  echo "[$(ts)] Starting python (log: $run_log)"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py 2>&1 | tee "$run_log"

  if [[ -d "logs/rl/a2c" ]]; then
    dest="logs/rl/a2c_lat_${safe_lat}"
    if [[ -e "$dest" ]]; then
      echo "ERROR: destination already exists: $dest"
      exit 1
    fi
    mv "logs/rl/a2c" "$dest"
  else
    echo "WARNING: logs/rl/a2c not found after run $lat"
  fi

  echo "[$(ts)] Finished A2C run for latency $lat"
  echo
done

echo "[$(ts)] All A2C latency runs completed."