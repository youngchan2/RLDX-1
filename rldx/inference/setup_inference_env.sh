#!/usr/bin/env bash
# Setup the RLDX inference venv (sm_120 / Blackwell / CUDA 13.x) from
# ``rldx/inference/pyproject.toml`` + ``uv.lock`` (torch 2.10.0+cu130).
#
# Adapted from Isaac-GR00T-AlinVLA/scripts/setup_inference_env.sh. This env is
# SEPARATE from the repo root ``.venv`` (torch 2.7/cu126).
#
# flash-attn is built from source; the GPU arch(es) it is compiled for are
# taken from the command line (default: 120). torch 2.10+cu130 itself supports
# sm_80/86/90/100/120, so the same venv can drive A100/H100/Blackwell — but the
# flash-attn kernels only run on the arch(es) you build here.
#
# Usage:
#   bash rldx/inference/setup_inference_env.sh [ARCH ...]
#
#   ARCH tokens:  80 (A100)  86  90 (H100/H200)  100  120 (Blackwell/RTX 5090)
#   Default:      120
#   Examples:
#     setup_inference_env.sh             # sm_120 only  (5090)
#     setup_inference_env.sh 90 120      # H100 + Blackwell
#     setup_inference_env.sh 80 90 120   # A100 + H100 + Blackwell
#   Also accepts a comma list (80,90,120) and the FLASH_ATTN_CUDA_ARCHS env var.
#
# Re-running with a different arch set rebuilds flash-attn (a plain ``uv sync``
# would otherwise reuse the cached wheel, since the version is unchanged).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
VENV_DIR="$SCRIPT_DIR/.venv"

# --- resolve flash-attn target arch(es) --------------------------------------
if [ "$#" -ge 1 ]; then
    ARCHS="$*"                       # space-separated args -> normalised below
elif [ -n "${FLASH_ATTN_CUDA_ARCHS:-}" ]; then
    ARCHS="$FLASH_ATTN_CUDA_ARCHS"
else
    ARCHS="120"
fi
ARCHS="${ARCHS//,/;}"; ARCHS="${ARCHS// /;}"   # accept ',' or ' ' as separators
# Validate every token and de-dup while preserving order.
SEEN=""; CLEAN=""
IFS=';' read -ra _toks <<< "$ARCHS"
for t in "${_toks[@]}"; do
    [ -n "$t" ] || continue
    case "$t" in
        80|86|90|100|120) ;;
        *) echo "ERROR: unsupported arch '$t' (use 80, 86, 90, 100, 120)." >&2; exit 1 ;;
    esac
    case ";$SEEN;" in *";$t;"*) continue ;; esac
    SEEN="$SEEN;$t"; CLEAN="${CLEAN:+$CLEAN;}$t"
done
[ -n "$CLEAN" ] || { echo "ERROR: no arch resolved." >&2; exit 1; }
export FLASH_ATTN_CUDA_ARCHS="$CLEAN"

# --- locate CUDA 13.x toolkit (nvcc) for the flash-attn source build ---------
if [ -z "${CUDA_HOME:-}" ] || [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    CUDA_HOME=""
    for d in $(ls -d /usr/local/cuda-13.* /usr/local/cuda-13 2>/dev/null | sort -Vr); do
        [ -x "$d/bin/nvcc" ] && { CUDA_HOME="$d"; break; }
    done
fi
[ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ] || {
    echo "ERROR: no CUDA 13.x toolkit (nvcc) found under /usr/local; flash-attn source build needs it." >&2
    exit 1
}
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

# --- rebuild flash-attn unless this venv was already built for these archs ----
# uv's build cache is keyed by the sdist, NOT by FLASH_ATTN_CUDA_ARCHS, so it
# will happily reuse a wheel built for a different arch. To get the requested
# arch we must purge the cached flash-attn wheel and force a fresh source build.
# The marker records the arch set this venv was *successfully verified* for.
MARKER="$VENV_DIR/.flash_attn_archs"
SYNC_ARGS=()
if [ -f "$MARKER" ] && [ "$(cat "$MARKER" 2>/dev/null)" = "$FLASH_ATTN_CUDA_ARCHS" ] \
   && [ -x "$VENV_DIR/bin/python" ]; then
    echo "flash-attn already built for sm_{$FLASH_ATTN_CUDA_ARCHS}; no rebuild needed."
else
    [ -f "$MARKER" ] && echo "flash-attn arch ($(cat "$MARKER")) != requested ($FLASH_ATTN_CUDA_ARCHS); rebuilding."
    echo "Purging cached flash-attn wheel to force a source build..."
    uv cache clean flash-attn >/dev/null 2>&1 || true
    SYNC_ARGS+=(--reinstall-package flash-attn)
fi

echo "=== uv sync (torch 2.10.0+cu130, flash-attn source build for sm_{${FLASH_ATTN_CUDA_ARCHS}}) ==="
echo "    CUDA_HOME=$CUDA_HOME"
echo "    FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS"
echo "    venv=$VENV_DIR"
uv sync "${SYNC_ARGS[@]}"

echo ""
echo "=== Installed versions ==="
RLDX_ARCHS="$FLASH_ATTN_CUDA_ARCHS" .venv/bin/python - <<'PY'
import os, torch, flash_attn, torchvision, triton
want = [t for t in os.environ["RLDX_ARCHS"].split(";") if t]
print(f"torch:       {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"flash-attn:  {flash_attn.__version__}  (requested sm_{', sm_'.join(want)})")
print(f"triton:      {triton.__version__}")
print(f"CUDA:        {torch.version.cuda}")
archs = torch.cuda.get_arch_list()
print(f"arch_list:   {archs}")
missing = [t for t in want if not any(t in a for a in archs)]
assert not missing, f"torch build lacks sm_{missing} — cannot drive those GPUs (arch_list={archs})."
print(f"torch supports requested arch(es): OK")

# Exercise the flash-attn kernel on every visible GPU whose arch we built for.
import flash_attn.flash_attn_interface as fi
ran = False
for i in range(torch.cuda.device_count()):
    maj, minr = torch.cuda.get_device_capability(i)
    tok = f"{maj}{minr}"
    name = torch.cuda.get_device_name(i)
    if tok not in want:
        print(f"  cuda:{i} sm_{tok} {name}: skipped (not in build set)")
        continue
    q = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16, device=f"cuda:{i}")
    try:
        o = fi.flash_attn_func(q, q, q); torch.cuda.synchronize(i)
        print(f"  cuda:{i} sm_{tok} {name}: flash-attn kernel OK {tuple(o.shape)}")
        ran = True
    except Exception as e:
        print(f"  cuda:{i} sm_{tok} {name}: FAILED — {type(e).__name__}: {e}")
        raise
if not ran:
    print("  (no visible GPU matched the build set — kernel not exercised)")
PY

# Record the verified arch set only after the checks above pass (set -e aborts
# the script first otherwise), so a failed build never leaves a false marker.
echo "$FLASH_ATTN_CUDA_ARCHS" > "$MARKER"
echo ""
echo "Done. flash-attn verified for sm_{$FLASH_ATTN_CUDA_ARCHS}."
