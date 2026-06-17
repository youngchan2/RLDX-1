"""Device-capability helper for arch-conditioned kernel autotuning.

Centralizes the ``torch.cuda`` introspection used to decide *which* Triton
autotune candidate sub-lists to compile per GPU, so a single kernel source
serves consumer Blackwell (sm_120), A100 (sm_80) and H100 (sm_90) without
hardcoding any one architecture. The actual tile/mma lowering is left to the
Triton backend (MMA on sm_80, WGMMA on sm_90) — we never emit arch-specific
primitives here.

Usage in a kernel module (built once at import, before compile/CUDA-graph
capture, so it is graph-safe):

    from utils.device_caps import is_server_class
    _CONFIGS = _BASE_CONFIGS + (_BIG_TILE_CONFIGS if is_server_class() else [])

Profiles are cached per device index. The default index follows
``torch.cuda.current_device()`` so it reflects the device the benchmark /
server actually selected (e.g. cuda:1 = H100 on this box), not cuda:0.
"""

from __future__ import annotations

import functools

import torch


# Compute-capability shorthands (major*10 + minor).
CC_A100 = 80
CC_H100 = 90
CC_BLACKWELL_CONSUMER = 120  # RTX 5090 / RTX PRO 6000 Blackwell

# Server-grade datacenter parts this build targets for the "general" path.
_SERVER_CCS = frozenset({CC_A100, CC_H100})


def _current_index(idx: int | None) -> int:
    if idx is not None:
        return idx
    return torch.cuda.current_device() if torch.cuda.is_available() else 0


@functools.lru_cache(maxsize=None)
def device_profile(idx: int) -> dict:
    """Return cached capability/SM/shared-memory facts for device ``idx``."""
    cap = torch.cuda.get_device_capability(idx)
    props = torch.cuda.get_device_properties(idx)
    return {
        "cc": cap[0] * 10 + cap[1],
        "sm_count": props.multi_processor_count,
        "smem_per_sm": getattr(props, "shared_memory_per_multiprocessor", None),
        "name": props.name,
    }


def get_profile(idx: int | None = None) -> dict:
    """Profile for ``idx`` (defaults to the current CUDA device)."""
    return device_profile(_current_index(idx))


def compute_capability(idx: int | None = None) -> int:
    return get_profile(idx)["cc"]


def sm_count(idx: int | None = None) -> int:
    return get_profile(idx)["sm_count"]


def is_server_class(idx: int | None = None) -> bool:
    """True on A100 (sm_80) / H100 (sm_90); False on consumer Blackwell etc."""
    return compute_capability(idx) in _SERVER_CCS
