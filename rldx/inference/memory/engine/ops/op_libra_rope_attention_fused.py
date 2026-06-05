"""Register mem::libra_fused_attention custom op.

Fused RoPE + Block-Causal Attention for TransformerMemory, backed by the Libra
FragTile CUDA kernel (libra_memory_attn_rope). Drop-in replacement for
mem::fused_attention (same signature + output shape), so the memory chain can
swap to it. No QK RMSNorm, no GQA. RoPE in bf16, matching eager.

The Libra kernel reads the fused QKV (M, 3*H*D) strided in-place (no transpose),
applies RoPE in-kernel, and returns (H, M, D); we reshape to (M, H*D) to match
the op contract (column layout head*head_dim + d, identical to mem::fused_attention).
"""

from __future__ import annotations

import os
import sys

import torch

# Make the Libra package importable (kernels.rldx.memory.*).
_LIBRA_ROOT = os.environ.get("LIBRA_ROOT", "/home/chani227/rldx/Libra")
if _LIBRA_ROOT not in sys.path:
    sys.path.insert(0, _LIBRA_ROOT)

# Fixed placement config for the fused kernel: (block_s, num_warps).
# RB=1/NS=1/QKRF=0/PVRF=0 are hardcoded inside the kernel; only these two vary.
# (64, 8) is fastest at the real VLA shape M=64.
# block_s/num_warps default to None → the wrapper autotunes over its compiled
# config set and caches the best per shape (set a tuple to pin a config).
_LIBRA_CFG = (None, None)

_fwd = None


def _get_libra_forward():
    """Lazily import + JIT-build the Libra fused-RoPE memory kernel (cached)."""
    global _fwd
    if _fwd is None:
        from kernels.rldx.memory.libra_memory_attn_rope_wrapper import (
            forward_libra_mem_rope_attn,
        )
        _fwd = forward_libra_mem_rope_attn
    return _fwd


@torch.library.custom_op("mem::libra_fused_attention", mutates_args=())
def libra_fused_memory_attention(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    signed_sin: torch.Tensor,
    num_heads: int,
    head_dim: int,
    block_attn_size: int,
) -> torch.Tensor:
    """RoPE + Block-Causal attention (Libra FragTile fused kernel).

    Args:
        qkv:             (M, 3*num_heads*head_dim) bf16 — fused QKV projection output
        cos:             (M, head_dim) bf16 — pre-computed RoPE cos
        signed_sin:      (M, head_dim) bf16 — pre-computed signed sin
        num_heads:       int (16)
        head_dim:        int (256)
        block_attn_size: int (16)

    Returns:
        (M, num_heads * head_dim) bf16 — attention output (before o_proj).
    """
    block_s, num_warps = _LIBRA_CFG
    fwd = _get_libra_forward()
    # (H, M, D) bf16
    out_hmd = fwd(qkv, cos, signed_sin, num_heads, block_attn_size, 1, block_s, num_warps)
    M = qkv.shape[0]
    # (H, M, D) -> (M, H, D) -> (M, H*D)  (column = head*D + d, matches Triton op)
    return out_hmd.permute(1, 0, 2).reshape(M, num_heads * head_dim).contiguous()


@libra_fused_memory_attention.register_fake
def _(qkv, cos, signed_sin, num_heads, head_dim, block_attn_size):
    M = qkv.shape[0]
    return qkv.new_empty((M, num_heads * head_dim))
