#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_ID=0
CONFIG_FILE="config.ini"

THROUGHPUTS=(100 150 200 250 300 350 400 450 500)

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
    BEGIN { in_section=0; changed=0 }

    /^\[/ {
      in_section = ($0 == "[" section "]")
    }

    in_section &&
    $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
      sub(/=.*/, "= " value)
      changed=1
    }

    { print }

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

make_heuristic_dirs() {
  mkdir -p logs/heuristic/{splits,system}
}

mkdir -p logs

echo "=================================================="
echo "Starting heuristic throughput sweep"
echo "N_EPISODES = 300"
echo "MAX_INFERENCE_LATENCY = 2.0"
echo "=================================================="

for throughput in "${THROUGHPUTS[@]}"; do

  destination="logs/heuristic_thr_${throughput}"

  echo
  echo "--------------------------------------------------"
  echo "[$(ts)] Throughput = $throughput MBps"
  echo "PARAM_PATH = $throughput"
  echo "N_EPISODES = 300"
  echo "MAX_INFERENCE_LATENCY = 2.0"
  echo "--------------------------------------------------"

  if [[ -e "$destination" ]]; then
    echo "ERROR: destination already exists: $destination"
    exit 1
  fi

  if [[ ! -d "input/episode_parameters/${throughput}" ]]; then
    echo "ERROR: parameter folder not found:"
    echo "input/episode_parameters/${throughput}"
    exit 1
  fi

  rm -rf logs/heuristic
  make_heuristic_dirs

  set_ini_value "ALGORITHM" "SPLIT_ALGORITHM" "6"
  set_ini_value "ALGORITHM" "N_EPISODES" "300"
  set_ini_value "ALGORITHM" "START_EPISODE" "1"
  set_ini_value "ALGORITHM" "PARAM_PATH" "$throughput"
  set_ini_value "ALGORITHM" "MAX_INFERENCE_LATENCY" "2.0"

  echo "[$(ts)] Configuration:"
  echo "  $(grep -m1 '^SPLIT_ALGORITHM' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^N_EPISODES' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PARAM_PATH' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^MAX_INFERENCE_LATENCY' "$CONFIG_FILE")"

  cp "$CONFIG_FILE" \
    "logs/config_heuristic_thr_${throughput}.ini"

  run_log="logs/output_heuristic_thr_${throughput}.log"

  CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python main.py 2>&1 | tee "$run_log"

  if [[ ! -d "logs/heuristic" ]]; then
    echo "ERROR: logs/heuristic was not created."
    exit 1
  fi

  mv logs/heuristic "$destination"

  echo "[$(ts)] Finished throughput = $throughput MBps"
  echo "Results: $destination"
done

echo
echo "=================================================="
echo "[$(ts)] All heuristic throughput runs completed."
echo "=================================================="