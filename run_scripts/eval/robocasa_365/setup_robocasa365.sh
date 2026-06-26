#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"

bash "$PROJECT_REPO/rldx/eval/sim/robocasa365/setup_RoboCasa365.sh"

# This sim venv runs only the gym/mujoco client (the model is served by
# the main RLDX venv over zmq), so flash-attn is never loaded. Disable
# it so transformers falls back to SDPA and avoid an ABI mismatch
# between the sim venv's torch and the prebuilt flash-attn wheel.
SITE="$PROJECT_REPO/rldx/eval/sim/robocasa365/robocasa365_uv/.venv/lib/python3.10/site-packages"
mv "$SITE/flash_attn" "$SITE/flash_attn__disabled" 2>/dev/null || true
mv "$SITE"/flash_attn_2_cuda*.so "$SITE/flash_attn_2_cuda__disabled.so" 2>/dev/null || true
mv "$SITE"/flash_attn_cuda*.so "$SITE/flash_attn_cuda__disabled.so" 2>/dev/null || true
