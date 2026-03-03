#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_ID=0
CONFIG_FILE="config.ini"
RUN_ID=2000

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "Running in: $(pwd)"
echo "Using python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "Config file: $CONFIG_FILE"
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

echo "=============================="
echo "[$(ts)] A2C single run (system params folder = $RUN_ID)"
echo "=============================="

# Clean any previous A2C folder
rm -rf logs/rl/a2c
make_a2c_dirs

# ---- Update config for A2C run ----
set_ini_value "ALGORITHM" "SPLIT_ALGORITHM" "2"      # RL-based split
set_ini_value "ALGORITHM" "RL_ALGORITHM" "2"         # 2 = A2C
set_ini_value "ALGORITHM" "PARAM_PATH" "$RUN_ID"     # read params from input/episode_parameters/2000/
# If you have something like SYSTEM_PARAMS_MBPS in config, set it here too.
# set_ini_value "ALGORITHM" "SYSTEM_PARAMS_MBPS" "2000"

set_ini_value "DRL_HYPERPARAMETERS" "LR_ACTOR" "0.0003"       # 3e-4
set_ini_value "DRL_HYPERPARAMETERS" "LR_CRITIC" "0.001"      # 1e-3
set_ini_value "DRL_HYPERPARAMETERS" "ENTROPY_FACTOR" "0.005"  # 5e-3

echo "[$(ts)] Config check:"
echo "  $(grep -m1 '^SPLIT_ALGORITHM' "$CONFIG_FILE")"
echo "  $(grep -m1 '^RL_ALGORITHM' "$CONFIG_FILE")"
echo "  $(grep -m1 '^PARAM_PATH' "$CONFIG_FILE")"
echo "  $(grep -m1 '^LR_ACTOR' "$CONFIG_FILE")"
echo "  $(grep -m1 '^LR_CRITIC' "$CONFIG_FILE")"
echo "  $(grep -m1 '^ENTROPY_FACTOR' "$CONFIG_FILE")"
echo

# Save exact config used
cp "$CONFIG_FILE" "logs/config_a2c_${RUN_ID}.ini"

run_log="logs/output_a2c_${RUN_ID}.log"
echo "[$(ts)] Starting python (log: $run_log)"
CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py 2>&1 | tee "$run_log"

# ---- Archive outputs ----
# If your code writes A2C logs into logs/rl/a2c already, keep them.
# If it still writes into logs/rl/ddqn (hardcoded), move them into a2c.
if [[ -d "logs/rl/ddqn" ]] && [[ ! -d "logs/rl/a2c/models" ]]; then
  echo "[$(ts)] Detected outputs in logs/rl/ddqn; moving to logs/rl/a2c"
  rm -rf logs/rl/a2c
  mv logs/rl/ddqn logs/rl/a2c
  make_a2c_dirs
fi

echo "[$(ts)] Finished A2C run ($RUN_ID)"