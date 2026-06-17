"""Register rldx_backbone::libra_vision_attention custom op.

Fused vision RoPE + NON-CAUSAL (cu_seqlens) attention, backed by the Libra
FragTile CUDA kernel (libra_vision_attn_rope). Drop-in replacement for
rldx_backbone::vision_attention (same signature + output shape), so the vision
chain can swap to it. head_dim=72 padded to D_pad=96 inside the kernel; RoPE
(rotate_half, fp32 cos/sin) is applied in-kernel; V is passthrough.

The Libra kernel reads the fused QKV (M, 3*H*D) strided in-place and returns
(H, M, D); we reshape to (M, H*D) to match the op contract. `scaling` is handled
internally by the kernel (1/sqrt(head_dim)); it is accepted for signature parity.
"""

from __future__ import annotations

import os
import sys

import torch

# Make the Libra package importable (kernels.rldx.vision.*).
_LIBRA_ROOT = os.environ.get("LIBRA_ROOT", "/home/chani227/rldx/Libra")
if _LIBRA_ROOT not in sys.path:
    sys.path.insert(0, _LIBRA_ROOT)

_fwd = None


def _get_libra_forward():
    """Lazily import + JIT-build the Libra fused-RoPE vision kernel (cached)."""
    global _fwd
    if _fwd is None:
        from kernels.rldx.vision.libra_vision_attn_rope_wrapper import (
            forward_libra_vis_rope_attn,
        )
        _fwd = forward_libra_vis_rope_attn
    return _fwd


@torch.library.custom_op("rldx_backbone::libra_vision_attention", mutates_args=())
def libra_vision_attention(
    qkv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scaling: float,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Fused vision RoPE + non-causal attention (Libra FragTile kernel).

    Args:
        qkv:        (M, 3 * num_heads * head_dim) bf16 — fused QKV
        rope_cos:   (M, head_dim) fp32 — baked cos (static)
        rope_sin:   (M, head_dim) fp32 — baked sin with rotate_half sign (static)
        cu_seqlens: (num_seqs+1,) int32 — per-image boundaries
        scaling:    float — head_dim ** -0.5 (applied internally by the kernel)
        num_heads:  int
        head_dim:   int

    Returns:
        (M, num_heads * head_dim) bf16 — attention output, column layout head*head_dim + d.
    """
    fwd = _get_libra_forward()
    out_hmd = fwd(qkv, rope_cos, rope_sin, cu_seqlens, num_heads, 1)   # (H, M, D), autotuned
    M = qkv.shape[0]
    # (H, M, D) -> (M, H, D) -> (M, H*D)  (column = head*D + d, matches Triton op)
    return out_hmd.permute(1, 0, 2).reshape(M, num_heads * head_dim).contiguous()


@libra_vision_attention.register_fake
def _(qkv, rope_cos, rope_sin, cu_seqlens, scaling, num_heads, head_dim):
    M = qkv.shape[0]
    return qkv.new_empty((M, num_heads * head_dim))
