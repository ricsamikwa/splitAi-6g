#!/usr/bin/env bash
set -euo pipefail

# Run from project root (where this script lives)
cd "$(dirname "$0")"

GPU_ID=0
CONFIG_FILE="config.ini"   # <-- change if your config is named differently

# The run IDs you want: 100, 150, ..., 500
RUN_IDS=(100 150 200 250 300 350 400 450 500)

echo "Running in: $(pwd)"
echo "Using python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "Config file: $CONFIG_FILE"
echo

# --- helper: set a key=value inside an INI section ---
# usage: set_ini_value "SECTION" "KEY" "VALUE"
set_ini_value () {
  local section="$1"
  local key="$2"
  local value="$3"

  # This replaces lines like KEY = something within the [SECTION] block.
  perl -0777 -i -pe '
    my ($sec,$key,$val) = @ARGV;
    s/(\['"$section"'\][^\[]*?\n\s*'"$key"'\s*=\s*).*?(\s*(?:\n|\r\n))/\1'"$value"'\2/s
      or die "Failed to set '"$key"' in section ['"$section"']\n";
  ' "$section" "$key" "$value" "$CONFIG_FILE"
}

# --- helper: (re)create ddqn subdirs ---
make_ddqn_dirs () {
  mkdir -p logs/rl/ddqn/{epsilon,loss,models,reward,splits,system}
}

# mkdir -p logs
make_ddqn_dirs

for rid in "${RUN_IDS[@]}"; do
  echo "=============================="
  echo "Run ID: $rid"
  echo "=============================="

  # 1) change parameters for this run
  # مثال: set N_EPISODES based on rid, or change epsilon schedule, etc.
  # Replace these with what you actually want to sweep.
  set_ini_value "ALGORITHM" "PARAM_PATH" "$rid"

  # If you want something to depend on rid, do it like:
  # set_ini_value "DRL_HYPERPARAMETERS" "LR" "$(python - <<PY
  # rid=$rid
  # print(1e-5)  # compute per rid if needed
  # PY
  # )"

  # 2) run main.py (blocking). Use tee to keep per-run stdout log.
  run_log="logs/output_${rid}.log"
  echo "Starting run $rid (log: $run_log)"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python main.py 2>&1 | tee "$run_log"

  # 3) archive ddqn folder
  if [[ -d "logs/rl/ddqn" ]]; then
    dest="logs/rl/ddqn${rid}"
    # If destination exists, don’t overwrite accidentally
    if [[ -e "$dest" ]]; then
      echo "ERROR: destination already exists: $dest"
      exit 1
    fi
    mv "logs/rl/ddqn" "$dest"
  else
    echo "WARNING: logs/rl/ddqn not found after run $rid"
  fi

  # 4) recreate fresh ddqn folder structure for next run
  make_ddqn_dirs

  echo "Finished run $rid"
  echo
done

echo "All runs completed."