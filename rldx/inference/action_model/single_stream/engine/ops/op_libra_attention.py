"""Register ss::fused_attention_2way custom op.

Triton kernel for RMSNorm + RoPE, then F.sdpa for attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import os, sys

_LIBRA_ROOT = os.environ.get("LIBRA_ROOT", "/home/chani227/rldx/Libra")
if _LIBRA_ROOT not in sys.path:
    sys.path.insert(0, _LIBRA_ROOT)

_ext = None
def _get_libra_ext():
    global _ext
    if _ext is None:
        from kernels.rldx.action.libra_action_attn_wrapper import _get_ext
        _ext = _get_ext()
    return _ext

_LIBRA_CFG = (32, 1, 0, 2, 4, 0, 0, 0, 2)  # autotuned: H=24 TOTAL=82 D=64 bf16 (1.89x vs sdpa)
_D = 64

@torch.library.custom_op("ss::libra_fused_attention_2way", mutates_args=())
def libra_fused_attention(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    n_sa: int,
) -> torch.Tensor:
    """RMSNorm + RoPE (Triton) + F.sdpa for SingleStreamBlock."""
    from single_stream.engine.kernels.rmsnorm_rope_ss import rmsnorm_rope_kernel

    M = qkv.shape[0]
    N, H, D = 1536, 24, 64

    q_out = torch.empty((H, M, D), device=qkv.device, dtype=torch.bfloat16)
    k_out = torch.empty((H, M, D), device=qkv.device, dtype=torch.bfloat16)
    v_out = torch.empty((H, M, D), device=qkv.device, dtype=torch.bfloat16)

    rope_cos = rope_cos.contiguous()
    rope_sin = rope_sin.contiguous()

    rmsnorm_rope_kernel[lambda meta: ((M + meta["BLOCK_S"] - 1) // meta["BLOCK_S"], H)](
        q_out,
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        k_out,
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        v_out,
        v_out.stride(0),
        v_out.stride(1),
        v_out.stride(2),
        qkv,
        qkv.stride(0),
        qkv.stride(1),
        q_norm_weight,
        k_norm_weight,
        rope_cos,
        rope_cos.stride(0),
        rope_cos.stride(1),
        rope_sin,
        rope_sin.stride(0),
        rope_sin.stride(1),
        BLOCK_S=128,
        BLOCK_N=D,
        D=D,
        H=H,
        M=M,
        N=N,
        N_SA=n_sa,
    )

    ext = _get_libra_ext()
    bs, rb, ns, nw, qkrf, pvrf, prf, ws, ab = _LIBRA_CFG
    attn_out = ext.forward_rldx_action_attn_cuda(
        q_out, k_out, v_out, H, _D, 1, bs, rb, ns, nw, qkrf, pvrf, prf, ws, ab,
    )

    return attn_out.permute(1, 0, 2).contiguous().view(1, M, N)

    # --- Alternative: fully fused Triton kernel (RMSNorm + RoPE + Attention) ---
    # from single_stream.engine.kernels.attention_fusion_ss import fused_rmsnorm_rope_attention_ss
    # k_norm = torch.empty((H, M, D), device=qkv.device, dtype=torch.float32)
    # o2 = torch.empty((M, N), device=qkv.device, dtype=torch.float32)
    # v = torch.empty((H, M, D), device=qkv.device, dtype=torch.float32)
    # fused_rmsnorm_rope_attention_ss[...](k_norm, o2, v, qkv, ...)
    # return o2.to(torch.bfloat16).view(1, M, N)


@libra_fused_attention.register_fake
def _(qkv, q_norm_weight, k_norm_weight, rope_cos, rope_sin, n_sa):
    M = qkv.shape[0]
    return qkv.new_empty((1, M, 1536))
