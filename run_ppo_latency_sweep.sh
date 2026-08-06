#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_ID=0
CONFIG_FILE="config.ini"
PPO_RL_ALGORITHM=3
PARAM_FOLDER=2000

LATENCIES=(
  0.225
  0.250
  0.275
  0.300
  0.325
  0.350
  0.375
  0.400
)

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

set_ini_value() {
  local section="$1"
  local key="$2"
  local value="$3"
  local tmp

  tmp="$(mktemp)"

  awk -v section="$section" -v key="$key" -v value="$value" '
    BEGIN {
      in_section = 0
      changed = 0
    }

    /^\[/ {
      in_section = ($0 == "[" section "]")
    }

    in_section &&
    $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
      sub(/=.*/, "= " value)
      changed = 1
    }

    {
      print
    }

    END {
      if (changed == 0) {
        print "ERROR: could not update " key \
              " in section [" section "]" > "/dev/stderr"
        exit 2
      }
    }
  ' "$CONFIG_FILE" > "$tmp"

  mv "$tmp" "$CONFIG_FILE"
}

make_ppo_dirs() {
  mkdir -p logs/rl/ppo/{epsilon,loss,models,reward,splits,system}
}

mkdir -p logs logs/rl

echo "Running in: $(pwd)"
echo "Using Python: $(which python)"
echo "GPU: $GPU_ID"
echo "System parameter folder: $PARAM_FOLDER"
echo

for latency in "${LATENCIES[@]}"; do
  safe_latency="${latency/./p}"
  destination="logs/rl/ppo_lat_${safe_latency}"

  echo "=================================================="
  echo "[$(ts)] PPO latency run"
  echo "MAX_INFERENCE_LATENCY = $latency"
  echo "=================================================="

  if [[ -e "$destination" ]]; then
    echo "ERROR: destination already exists: $destination"
    exit 1
  fi

  rm -rf logs/rl/ppo
  make_ppo_dirs

  set_ini_value "ALGORITHM" "SPLIT_ALGORITHM" "2"
  set_ini_value "ALGORITHM" "RL_ALGORITHM" "$PPO_RL_ALGORITHM"
  set_ini_value "ALGORITHM" "INFERENCE" "0"
  set_ini_value "ALGORITHM" "N_EPISODES" "10000"
  set_ini_value "ALGORITHM" "START_EPISODE" "1"
  set_ini_value "ALGORITHM" "MAX_ENERGY_CREDIT" "90"
  set_ini_value "ALGORITHM" "ACCURACY_DECREASE" "80"
  set_ini_value "ALGORITHM" "PARAM_PATH" "$PARAM_FOLDER"
  set_ini_value "ALGORITHM" "MAX_INFERENCE_LATENCY" "$latency"

  set_ini_value \
    "DRL_HYPERPARAMETERS" \
    "PPO_CLIP_EPSILON" \
    "0.2"

  set_ini_value \
    "DRL_HYPERPARAMETERS" \
    "PPO_GAE_LAMBDA" \
    "0.95"

  set_ini_value \
    "DRL_HYPERPARAMETERS" \
    "PPO_EPOCHS" \
    "4"

  echo "[$(ts)] Configuration:"
  echo "  $(grep -m1 '^MAX_INFERENCE_LATENCY' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PARAM_PATH' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^N_EPISODES' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PPO_CLIP_EPSILON' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PPO_GAE_LAMBDA' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PPO_EPOCHS' "$CONFIG_FILE")"
  echo

  cp \
    "$CONFIG_FILE" \
    "logs/config_ppo_lat_${safe_latency}.ini"

  run_log="logs/output_ppo_lat_${safe_latency}.log"

  CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python main.py 2>&1 | tee "$run_log"

  if [[ ! -d "logs/rl/ppo" ]]; then
    echo "ERROR: logs/rl/ppo was not created."
    exit 1
  fi

  mv logs/rl/ppo "$destination"

  echo "[$(ts)] Saved results to $destination"
  echo
done

echo "[$(ts)] All PPO latency experiments completed."