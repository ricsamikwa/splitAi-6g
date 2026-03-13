#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_ID=0
CONFIG_FILE="config.ini"
PARAM_FOLDER=2000
WEIGHTS=(0.9 0.5)

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

for w in "${WEIGHTS[@]}"; do
  safe_w="${w/./p}"

  echo "=============================="
  echo "[$(ts)] DDQN run: WEIGHT_INFERENCE_TIME=$w"
  echo "=============================="

  rm -rf logs/rl/ddqn
  make_ddqn_dirs

  # DDQN config
  set_ini_value "ALGORITHM" "SPLIT_ALGORITHM" "2"
  set_ini_value "ALGORITHM" "RL_ALGORITHM" "1"
  set_ini_value "ALGORITHM" "PARAM_PATH" "$PARAM_FOLDER"
  set_ini_value "ALGORITHM" "MAX_INFERENCE_LATENCY" "2.0"
  set_ini_value "ALGORITHM" "WEIGHT_INFERENCE_TIME" "$w"

  # Same DDQN learning params as before
  set_ini_value "DRL_HYPERPARAMETERS" "LR" "0.00001"
  set_ini_value "DRL_HYPERPARAMETERS" "TAU" "0.001"

  echo "[$(ts)] Config check:"
  echo "  $(grep -m1 '^SPLIT_ALGORITHM' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^RL_ALGORITHM' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PARAM_PATH' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^MAX_INFERENCE_LATENCY' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^WEIGHT_INFERENCE_TIME' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^LR' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^TAU' "$CONFIG_FILE")"
  echo

  cp "$CONFIG_FILE" "logs/config_ddqn_weight_${safe_w}.ini"
  run_log="logs/output_ddqn_weight_${safe_w}.log"

  echo "[$(ts)] Starting python (log: $run_log)"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py 2>&1 | tee "$run_log"

  if [[ -d "logs/rl/ddqn" ]]; then
    dest="logs/rl/ddqn_weight_${safe_w}"
    if [[ -e "$dest" ]]; then
      echo "ERROR: destination already exists: $dest"
      exit 1
    fi
    mv "logs/rl/ddqn" "$dest"
  else
    echo "WARNING: logs/rl/ddqn not found after run $w"
  fi

  echo "[$(ts)] Finished DDQN run for WEIGHT_INFERENCE_TIME=$w"
  echo
done

echo "[$(ts)] All DDQN weight runs completed."