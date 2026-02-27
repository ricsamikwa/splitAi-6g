#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_ID=0
CONFIG_FILE="config.ini"
RUN_IDS=(100 150 200 250 300 350 400 450 500)

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
    /^\[/ {
      in_section = ($0 == "[" section "]")
    }
    in_section && $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
      sub(/=.*/, "= " value)
      changed=1
    }
    { print }
    END {
      if (changed == 0) {
        # exit non-zero so bash (set -e) stops with a clear error
        exit 2
      }
    }
  ' "$CONFIG_FILE" > "$tmp"

  mv "$tmp" "$CONFIG_FILE"
}

make_ddqn_dirs () {
  mkdir -p logs/rl/ddqn/{epsilon,loss,models,reward,splits,system}
}

mkdir -p logs logs/rl

for rid in "${RUN_IDS[@]}"; do
  echo "=============================="
  echo "[$(ts)] Run ID: $rid"
  echo "=============================="

  # Ensure clean per-run folder
  rm -rf logs/rl/ddqn
  make_ddqn_dirs

  # Update config for this run
  set_ini_value "ALGORITHM" "PARAM_PATH" "$rid"

  # Save the exact config used for reproducibility
  cp "$CONFIG_FILE" "logs/config_${rid}.ini"

  run_log="logs/output_${rid}.log"
  echo "[$(ts)] Starting python (log: $run_log)"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py 2>&1 | tee "$run_log"

  # Archive outputs
  if [[ -d "logs/rl/ddqn" ]]; then
    dest="logs/rl/ddqn${rid}"
    if [[ -e "$dest" ]]; then
      echo "ERROR: destination already exists: $dest"
      exit 1
    fi
    mv "logs/rl/ddqn" "$dest"
  else
    echo "WARNING: logs/rl/ddqn not found after run $rid"
  fi

  echo "[$(ts)] Finished run $rid"
  echo
done

echo "[$(ts)] All runs completed."