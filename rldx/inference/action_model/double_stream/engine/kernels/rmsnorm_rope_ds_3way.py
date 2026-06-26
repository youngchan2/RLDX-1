"""RMSNorm + RoPE kernel for ExpandedDoubleStreamBlock (3-way: VL|SA|P).

Outputs Q/K/V as (H, M, D) bf16, ready for F.scaled_dot_product_attention.
Layout: rows [0, N_VL) = VL (no RoPE), [N_VL, N_VL+N_SA) = SA, [N_VL+N_SA, M) = P.

RoPE is fused in registers (reshape/split/interleave) — Q/K stored once each.

BLOCK_S is autotuned (pure pointwise kernel; occupancy is the only knob —
grid = (ceil(TOTAL/BLOCK_S), H)).
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": bs}, num_stages=ns, num_warps=nw)
        for bs in [8, 16, 32, 64]
        for ns in [1, 2]
        for nw in [2, 4, 8]
    ],
    key=["M", "N_VL", "N_SA", "N_P"],
)
@triton.jit
def rmsnorm_rope_kernel_3way(
    # Output buffers (H, M, D) bf16
    Q_out_ptr,
    Q_out_stride0: tl.constexpr,
    Q_out_stride1: tl.constexpr,
    Q_out_stride2: tl.constexpr,
    K_out_ptr,
    K_out_stride0: tl.constexpr,
    K_out_stride1: tl.constexpr,
    K_out_stride2: tl.constexpr,
    V_out_ptr,
    V_out_stride0: tl.constexpr,
    V_out_stride1: tl.constexpr,
    V_out_stride2: tl.constexpr,
    # Input QKV (3 separate buffers)
    SA_QKV_ptr,
    SA_QKV_stride0: tl.constexpr,
    SA_QKV_stride1: tl.constexpr,
    VL_QKV_ptr,
    VL_QKV_stride0: tl.constexpr,
    VL_QKV_stride1: tl.constexpr,
    P_QKV_ptr,
    P_QKV_stride0: tl.constexpr,
    P_QKV_stride1: tl.constexpr,
    # RMSNorm weights (6 buffers: Q/K for SA, VL, P)
    Q_norm_sa_ptr,
    K_norm_sa_ptr,
    Q_norm_vl_ptr,
    K_norm_vl_ptr,
    Q_norm_p_ptr,
    K_norm_p_ptr,
    # RoPE tables — SA
    SA_ROPE_COS_ptr,
    SA_ROPE_COS_stride0: tl.constexpr,
    SA_ROPE_COS_stride1: tl.constexpr,
    SA_ROPE_SIN_ptr,
    SA_ROPE_SIN_stride0: tl.constexpr,
    SA_ROPE_SIN_stride1: tl.constexpr,
    # RoPE tables — P
    P_ROPE_COS_ptr,
    P_ROPE_COS_stride0: tl.constexpr,
    P_ROPE_COS_stride1: tl.constexpr,
    P_ROPE_SIN_ptr,
    P_ROPE_SIN_stride0: tl.constexpr,
    P_ROPE_SIN_stride1: tl.constexpr,
    # Constants
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    N_VL: tl.constexpr,
    N_SA: tl.constexpr,
    N_P: tl.constexpr,
):
    """RMSNorm + RoPE for 3-way [VL | SA | P]. Outputs Q/K/V as (H, M, D) bf16."""
    s = tl.program_id(0) * BLOCK_S
    h = tl.program_id(1)
    rs = s + tl.arange(0, BLOCK_S)
    rd = tl.arange(0, BLOCK_N)
    q_col = h * D
    k_col = N + h * D
    v_col = 2 * N + h * D
    s_mask = rs < M
    d_mask = rd < D
    mask_v = s_mask[:, None] & d_mask[None, :]

    # --- Region classification ---
    is_vl = rs < N_VL
    is_sa = (rs >= N_VL) & (rs < N_VL + N_SA)
    is_p = rs >= N_VL + N_SA
    # Per-region indices clamp to 0 for out-of-region lanes: a masked-out lane
    # still computes its load address, and on Blackwell an OOB address raises
    # cudaErrorIllegalAddress instead of being dropped.
    vl_idx = tl.where(is_vl & s_mask, rs, 0)
    sa_idx = tl.where(is_sa & s_mask, rs - N_VL, 0)
    p_idx = tl.where(is_p & s_mask, rs - N_VL - N_SA, 0)
    rs_safe = tl.where(s_mask, rs, M - 1)
    rd_safe = tl.where(d_mask, rd, D - 1)

    # --- Phase 1: Load QKV from 3 separate buffers ---
    Q_vl = tl.load(
        VL_QKV_ptr + vl_idx[:, None] * VL_QKV_stride0 + (q_col + rd_safe)[None, :] * VL_QKV_stride1,
        mask=is_vl[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    K_vl = tl.load(
        VL_QKV_ptr + vl_idx[:, None] * VL_QKV_stride0 + (k_col + rd_safe)[None, :] * VL_QKV_stride1,
        mask=is_vl[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    V_vl = tl.load(
        VL_QKV_ptr + vl_idx[:, None] * VL_QKV_stride0 + (v_col + rd_safe)[None, :] * VL_QKV_stride1,
        mask=is_vl[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    Q_sa = tl.load(
        SA_QKV_ptr + sa_idx[:, None] * SA_QKV_stride0 + (q_col + rd_safe)[None, :] * SA_QKV_stride1,
        mask=is_sa[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    K_sa = tl.load(
        SA_QKV_ptr + sa_idx[:, None] * SA_QKV_stride0 + (k_col + rd_safe)[None, :] * SA_QKV_stride1,
        mask=is_sa[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    V_sa = tl.load(
        SA_QKV_ptr + sa_idx[:, None] * SA_QKV_stride0 + (v_col + rd_safe)[None, :] * SA_QKV_stride1,
        mask=is_sa[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    Q_p = tl.load(
        P_QKV_ptr + p_idx[:, None] * P_QKV_stride0 + (q_col + rd_safe)[None, :] * P_QKV_stride1,
        mask=is_p[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    K_p = tl.load(
        P_QKV_ptr + p_idx[:, None] * P_QKV_stride0 + (k_col + rd_safe)[None, :] * P_QKV_stride1,
        mask=is_p[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    V_p = tl.load(
        P_QKV_ptr + p_idx[:, None] * P_QKV_stride0 + (v_col + rd_safe)[None, :] * P_QKV_stride1,
        mask=is_p[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Merge: 3-way select
    Q1 = tl.where(is_vl[:, None], Q_vl, tl.where(is_sa[:, None], Q_sa, Q_p))
    K1 = tl.where(is_vl[:, None], K_vl, tl.where(is_sa[:, None], K_sa, K_p))
    V1 = tl.where(is_vl[:, None], V_vl, tl.where(is_sa[:, None], V_sa, V_p))

    # --- Phase 2: Store V (directly to bf16) ---
    tl.store(
        V_out_ptr
        + h * V_out_stride0
        + rs_safe[:, None] * V_out_stride1
        + rd_safe[None, :] * V_out_stride2,
        V1.to(tl.bfloat16),
        mask=mask_v,
    )

    # --- Phase 3: QK RMSNorm (3-way weight select) ---
    eps = 1e-6
    q_weight_sa = tl.load(Q_norm_sa_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    k_weight_sa = tl.load(K_norm_sa_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    q_weight_vl = tl.load(Q_norm_vl_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    k_weight_vl = tl.load(K_norm_vl_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    q_weight_p = tl.load(Q_norm_p_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)
    k_weight_p = tl.load(K_norm_p_ptr + rd, mask=rd < D, other=1.0).to(tl.float32)

    q_weight = tl.where(
        is_vl[:, None],
        q_weight_vl[None, :],
        tl.where(is_sa[:, None], q_weight_sa[None, :], q_weight_p[None, :]),
    )
    k_weight = tl.where(
        is_vl[:, None],
        k_weight_vl[None, :],
        tl.where(is_sa[:, None], k_weight_sa[None, :], k_weight_p[None, :]),
    )

    Q_rms_inv = tl.rsqrt(tl.sum(Q1 * Q1, axis=1) / D + eps)
    Q_norm = (Q1 * Q_rms_inv[:, None]).to(tl.bfloat16) * q_weight.to(tl.bfloat16)
    K_rms_inv = tl.rsqrt(tl.sum(K1 * K1, axis=1) / D + eps)
    K_norm_val = (K1 * K_rms_inv[:, None]).to(tl.bfloat16) * k_weight.to(tl.bfloat16)

    # --- Phase 4 (single-pass): RoPE in registers, single store ---
    rd2 = tl.arange(0, D // 2)
    half_mask = rd2[None, :] < (D // 2)

    cos = tl.full([BLOCK_S, D // 2], 1.0, tl.float32)
    sin = tl.zeros([BLOCK_S, D // 2], tl.float32)

    sa_rope_idx = tl.where(is_sa & s_mask, rs - N_VL, 0)
    sa_m = is_sa[:, None] & half_mask
    cos_sa = tl.load(
        SA_ROPE_COS_ptr
        + sa_rope_idx[:, None] * SA_ROPE_COS_stride0
        + rd2[None, :] * SA_ROPE_COS_stride1,
        mask=sa_m,
        other=1.0,
    ).to(tl.float32)
    sin_sa = tl.load(
        SA_ROPE_SIN_ptr
        + sa_rope_idx[:, None] * SA_ROPE_SIN_stride0
        + rd2[None, :] * SA_ROPE_SIN_stride1,
        mask=sa_m,
        other=0.0,
    ).to(tl.float32)
    cos = tl.where(is_sa[:, None], cos_sa, cos)
    sin = tl.where(is_sa[:, None], sin_sa, sin)

    p_rope_idx = tl.where(is_p & s_mask, rs - N_VL - N_SA, 0)
    p_m = is_p[:, None] & half_mask
    cos_p = tl.load(
        P_ROPE_COS_ptr
        + p_rope_idx[:, None] * P_ROPE_COS_stride0
        + rd2[None, :] * P_ROPE_COS_stride1,
        mask=p_m,
        other=1.0,
    ).to(tl.float32)
    sin_p = tl.load(
        P_ROPE_SIN_ptr
        + p_rope_idx[:, None] * P_ROPE_SIN_stride0
        + rd2[None, :] * P_ROPE_SIN_stride1,
        mask=p_m,
        other=0.0,
    ).to(tl.float32)
    cos = tl.where(is_p[:, None], cos_p, cos)
    sin = tl.where(is_p[:, None], sin_p, sin)

    q_e, q_o = tl.split(tl.reshape(Q_norm, [BLOCK_S, D // 2, 2]))
    qe = q_e.to(tl.float32)
    qo = q_o.to(tl.float32)
    Q_roped = tl.interleave(
        (qe * cos - qo * sin).to(tl.bfloat16), (qe * sin + qo * cos).to(tl.bfloat16)
    )
    k_e, k_o = tl.split(tl.reshape(K_norm_val, [BLOCK_S, D // 2, 2]))
    ke = k_e.to(tl.float32)
    ko = k_o.to(tl.float32)
    K_roped = tl.interleave(
        (ke * cos - ko * sin).to(tl.bfloat16), (ke * sin + ko * cos).to(tl.bfloat16)
    )

    tl.store(
        Q_out_ptr
        + h * Q_out_stride0
        + rs_safe[:, None] * Q_out_stride1
        + rd_safe[None, :] * Q_out_stride2,
        Q_roped,
        mask=mask_v,
    )
    tl.store(
        K_out_ptr
        + h * K_out_stride0
        + rs_safe[:, None] * K_out_stride1
        + rd_safe[None, :] * K_out_stride2,
        K_roped,
        mask=mask_v,
    )
