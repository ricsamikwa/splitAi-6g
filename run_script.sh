#!/bin/bash
set -e

# Always run from the project root (where this script lives)
cd "$(dirname "$0")"

mkdir -p logs

gpu_id=0

echo "Running in: $(pwd)"
echo "Using python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=$gpu_id"

CUDA_VISIBLE_DEVICES=$gpu_id nohup python main.py > logs/output.log 2>&1 &

echo "Started PID: $!"
