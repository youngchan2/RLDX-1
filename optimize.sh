#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"
LOG_DIR="$PWD/log"
mkdir -p "$LOG_DIR"

# Repo root has no .venv; use the inference venv (torch 2.10/cu130).
PY=.venv/bin/python

run() {
  local name=$1 script=$2; shift 2
  echo "=== $name ==="
  "$PY" "$script" "$@" > "$LOG_DIR/$name.txt" 2>&1
  echo "  exit=$? → $LOG_DIR/$name.txt"
}

# run backbone      rldx/inference/backbone/benchmark_backbone.py
# run llm           rldx/inference/backbone/llm/benchmark_llm_chain.py
# run vision        rldx/inference/backbone/vision_encoder/benchmark_vision_chain.py
# run memory        rldx/inference/memory/benchmark_memory.py
# run action_model  rldx/inference/action_model/benchmark_action_model.py

run pretrain rldx/inference/benchmark_vla.py --model-type rldx_1_pretrain
run allex rldx/inference/benchmark_vla.py --model-type rldx_1_midtrain_allex --action-horizon 40
run droid rldx/inference/benchmark_vla.py --model-type rldx_1_midtrain_droid