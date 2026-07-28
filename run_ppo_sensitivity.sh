#!/usr/bin/env bash
set -euo pipefail

# Always run from the project root
cd "$(dirname "$0")"

GPU_ID=0
CONFIG_FILE="config.ini"

# Confirm these against the updated repository/config comments.
PPO_RL_ALGORITHM=3
PARAM_FOLDER=2000

ts() {
  date "+%Y-%m-%d %H:%M:%S"
}

echo "Running in: $(pwd)"
echo "Using Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "Config file: $CONFIG_FILE"
echo "PPO RL algorithm ID: $PPO_RL_ALGORITHM"
echo "System parameter folder: $PARAM_FOLDER"
echo

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

# Format:
# experiment_name|clip_epsilon|gae_lambda|epochs
EXPERIMENTS=(
  "default|0.2|0.95|4"
  "clip_0p1|0.1|0.95|4"
  "clip_0p3|0.3|0.95|4"
  "lambda_0p90|0.2|0.90|4"
  "lambda_0p99|0.2|0.99|4"
  "epochs_3|0.2|0.95|3"
  "epochs_8|0.2|0.95|8"
)

for experiment in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r run_name clip_epsilon gae_lambda ppo_epochs \
    <<< "$experiment"

  echo "=================================================="
  echo "[$(ts)] PPO experiment: $run_name"
  echo "  PPO_CLIP_EPSILON = $clip_epsilon"
  echo "  PPO_GAE_LAMBDA   = $gae_lambda"
  echo "  PPO_EPOCHS       = $ppo_epochs"
  echo "=================================================="

  destination="logs/rl/ppo_${run_name}"

  if [[ -e "$destination" ]]; then
    echo "ERROR: destination already exists: $destination"
    echo "Move or delete it before rerunning this experiment."
    exit 1
  fi

  # Start each experiment with an empty PPO output directory.
  rm -rf logs/rl/ppo
  make_ppo_dirs

  # Common experiment configuration.
  set_ini_value "ALGORITHM" "SPLIT_ALGORITHM" "2"
  set_ini_value "ALGORITHM" "RL_ALGORITHM" "$PPO_RL_ALGORITHM"
  set_ini_value "ALGORITHM" "INFERENCE" "0"
  set_ini_value "ALGORITHM" "N_EPISODES" "5000"
  set_ini_value "ALGORITHM" "START_EPISODE" "1"
  set_ini_value "ALGORITHM" "MAX_ENERGY_CREDIT" "90"
  set_ini_value "ALGORITHM" "PARAM_PATH" "$PARAM_FOLDER"

  # PPO sensitivity parameters.
  set_ini_value \
    "DRL_HYPERPARAMETERS" \
    "PPO_CLIP_EPSILON" \
    "$clip_epsilon"

  set_ini_value \
    "DRL_HYPERPARAMETERS" \
    "PPO_GAE_LAMBDA" \
    "$gae_lambda"

  set_ini_value \
    "DRL_HYPERPARAMETERS" \
    "PPO_EPOCHS" \
    "$ppo_epochs"

  echo "[$(ts)] Configuration check:"
  echo "  $(grep -m1 '^SPLIT_ALGORITHM' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^RL_ALGORITHM' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^N_EPISODES' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^MAX_ENERGY_CREDIT' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PARAM_PATH' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PPO_CLIP_EPSILON' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PPO_GAE_LAMBDA' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PPO_EPOCHS' "$CONFIG_FILE")"
  echo

  # Preserve the exact configuration used for reproducibility.
  config_copy="logs/config_ppo_${run_name}.ini"
  cp "$CONFIG_FILE" "$config_copy"

  run_log="logs/output_ppo_${run_name}.log"

  echo "[$(ts)] Starting PPO run"
  echo "Output log: $run_log"

  CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python main.py 2>&1 | tee "$run_log"

  if [[ ! -d "logs/rl/ppo" ]]; then
    echo "ERROR: expected PPO output folder was not created:"
    echo "  logs/rl/ppo"
    exit 1
  fi

  mv "logs/rl/ppo" "$destination"

  echo "[$(ts)] Finished PPO experiment: $run_name"
  echo "Results saved to: $destination"
  echo
done

echo "[$(ts)] All PPO sensitivity experiments completed."