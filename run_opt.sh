#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_ID=0
CONFIG_FILE="config.ini"
PARAM_FOLDER=2000

# 0.200, 0.225, ..., 0.400
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

make_opt_dirs () {
  mkdir -p logs/optimum/{splits,system}
}

mkdir -p logs

for lat in "${LATENCIES[@]}"; do
  echo "=============================="
  echo "[$(ts)] OPT run: MAX_INFERENCE_LATENCY=$lat"
  echo "=============================="

  # Clean OPT output directories so each run is isolated
  rm -rf logs/optimum
  make_opt_dirs

  # --- Set OPT + fixed params ---
  set_ini_value "ALGORITHM" "SPLIT_ALGORITHM" "3"          # 3 = optimal
  set_ini_value "ALGORITHM" "RL_ALGORITHM" "1"             # irrelevant for OPT, but keep stable
  set_ini_value "ALGORITHM" "INFERENCE" "0"                # training mode (as you used before)
  set_ini_value "ALGORITHM" "N_EPISODES" "9"
  set_ini_value "ALGORITHM" "ACCURACY_DECREASE" "80"
  set_ini_value "ALGORITHM" "PARAM_PATH" "$PARAM_FOLDER"   # system params 2000 MBps folder
  set_ini_value "ALGORITHM" "MAX_INFERENCE_LATENCY" "$lat"

  echo "[$(ts)] Config check:"
  echo "  $(grep -m1 '^SPLIT_ALGORITHM' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^MAX_INFERENCE_LATENCY' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^ACCURACY_DECREASE' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^N_EPISODES' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PARAM_PATH' "$CONFIG_FILE")"
  echo

  # Save config for this run
  safe_lat="${lat/./p}"   # 0.200 -> 0p200 (safe folder name)
  cp "$CONFIG_FILE" "logs/config_opt_lat_${safe_lat}.ini"

  run_log="logs/output_opt_lat_${safe_lat}.log"
  echo "[$(ts)] Starting python (log: $run_log)"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py 2>&1 | tee "$run_log"

  # Archive OPT outputs
  dest="logs/optimum_lat_${safe_lat}"
  if [[ -e "$dest" ]]; then
    echo "ERROR: destination already exists: $dest"
    exit 1
  fi
  mv logs/optimum "$dest"

  echo "[$(ts)] Finished OPT run (lat=$lat). Saved to $dest"
  echo
done

echo "[$(ts)] All OPT latency runs completed."