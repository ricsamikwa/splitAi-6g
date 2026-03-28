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

make_heuristic_dirs () {
  mkdir -p logs/heuristic/{splits,system}
}

mkdir -p logs

for rid in "${RUN_IDS[@]}"; do
  echo "=============================="
  echo "[$(ts)] Heuristic run: PARAM_PATH=$rid, MAX_INFERENCE_LATENCY=2.0"
  echo "=============================="

  rm -rf logs/heuristic
  make_heuristic_dirs

  set_ini_value "ALGORITHM" "SPLIT_ALGORITHM" "6"
  set_ini_value "ALGORITHM" "N_EPISODES" "50"
  set_ini_value "ALGORITHM" "PARAM_PATH" "$rid"
  set_ini_value "ALGORITHM" "MAX_INFERENCE_LATENCY" "2.0"

  echo "[$(ts)] Config check:"
  echo "  $(grep -m1 '^SPLIT_ALGORITHM' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^N_EPISODES' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^PARAM_PATH' "$CONFIG_FILE")"
  echo "  $(grep -m1 '^MAX_INFERENCE_LATENCY' "$CONFIG_FILE")"
  echo

  cp "$CONFIG_FILE" "logs/config_heuristic_thr_${rid}.ini"
  run_log="logs/output_heuristic_thr_${rid}.log"

  echo "[$(ts)] Starting python (log: $run_log)"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py 2>&1 | tee "$run_log"

  dest="logs/heuristic_thr_${rid}"
  if [[ -e "$dest" ]]; then
    echo "ERROR: destination already exists: $dest"
    exit 1
  fi
  mv logs/heuristic "$dest"

  echo "[$(ts)] Finished heuristic throughput run $rid"
  echo
done

echo "[$(ts)] All heuristic throughput runs completed."