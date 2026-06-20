"""Vision attention micro-benchmark: fused Triton kernel vs torch SDPA + separate ops.

Compares the single fused ``rldx_backbone::vision_attention`` Triton kernel
(baked RoPE + non-causal per-image attention in one launch) against an
equivalent path that performs the *same* computation but split into separate ops:

  - RoPE on Q/K   — plain PyTorch (eager) ops
  - non-causal per-image attention — F.scaled_dot_product_attention (dense batch)

This SDPA path mirrors ``CustomVisionEncoderChain._sdpa_attention`` (the server
path the real model already uses on A100/H100), so it is numerically faithful to
the fused kernel.

Both paths read the same fused QKV buffer ``(M, 3*H*D)`` and pre-computed
``rope_cos`` / ``rope_sin`` ``(M, D)`` fp32 (rotate_half sign folded into sin).
Equal-length per-image seqs reshape to a dense ``(num_seqs, H, S, D)`` batch.

Shapes follow the real vision encoder (H=16, D=72). The fused kernel
auto-dispatches direct (M>=128) vs split-KV; this bench forces DIRECT for the
baseline (the split path has a known correctness drift in the original).

Usage & Results:
    H100
    python inference/backbone/vision_encoder/benchmark_attention_sdpa.py
    ==============================================================================
    Vision attention  |  H=16 M=256 D=72 seq_len=64 num_seqs=4  bf16
    ==============================================================================
    Fused Triton kernel        :    35.23 us
    SDPA + separate RoPE ops   :    99.14 us
    speedup (sdpa / fused)     : 2.81x
    rel error (fused vs sdpa)  : 1.712e-03
    ==============================================================================
    python inference/backbone/vision_encoder/benchmark_attention_sdpa.py --cuda-graph
    ==============================================================================
    Vision attention  |  H=16 M=256 D=72 seq_len=64 num_seqs=4  bf16
    ==============================================================================
    Fused Triton kernel        :    35.28 us
    SDPA + separate RoPE ops   :    97.03 us
    speedup (sdpa / fused)     : 2.75x
    rel error (fused vs sdpa)  : 1.712e-03
    ------------------------------------------------------------------------------
    Fused   + CUDA graph       :    35.35 us  (1.00x vs eager)
    SDPA    + CUDA graph       :    40.09 us  (2.42x vs eager)
    speedup (sdpa / fused)     : 1.13x
    rel error (fused vs sdpa)  : 1.712e-03
    ==============================================================================
"""

from __future__ import annotations

import argparse
import importlib.util
import os

import torch
import torch.nn.functional as F


_KERNEL_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "engine",
    "kernels",
    "fused_vision_attention.py",
)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _time(fn, warmup=30, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def _capture(fn, warmup=5):
    """Capture fn() as a CUDA graph and return (graph, static_output).

    Warmup runs on a side stream first so any Triton autotune (which syncs and
    times) resolves BEFORE capture — autotuning cannot happen inside a capture.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def _signed_sin(sin, head_dim):
    sign = torch.ones(head_dim, device=sin.device, dtype=sin.dtype)
    sign[: head_dim // 2] = -1.0
    return (sign.unsqueeze(0) * sin).contiguous()


def sdpa_path(qkv, rope_cos, rope_sin, scaling, num_heads, head_dim, num_seqs):
    """Baked-RoPE (eager, fp32) + non-causal per-image attention via F.sdpa.

    Mirrors CustomVisionEncoderChain._sdpa_attention. ``rope_sin`` has the
    rotate_half sign baked in, so RoPE is q*cos + swap_half(q)*rope_sin where
    swap_half swaps the two head halves WITHOUT negation.
    """
    M = qkv.shape[0]
    qd = num_heads * head_dim
    S = M // num_seqs
    half = head_dim // 2

    q = qkv[:, :qd].reshape(M, num_heads, head_dim).float()
    k = qkv[:, qd : 2 * qd].reshape(M, num_heads, head_dim).float()
    v = qkv[:, 2 * qd : 3 * qd].reshape(M, num_heads, head_dim)

    cos = rope_cos.view(M, 1, head_dim)
    sin = rope_sin.view(M, 1, head_dim)
    q_sw = torch.cat((q[..., half:], q[..., :half]), dim=-1)
    k_sw = torch.cat((k[..., half:], k[..., :half]), dim=-1)
    q = (q * cos + q_sw * sin).to(v.dtype)
    k = (k * cos + k_sw * sin).to(v.dtype)

    q = q.reshape(num_seqs, S, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(num_seqs, S, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(num_seqs, S, num_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, scale=scaling)
    return out.transpose(1, 2).reshape(M, qd)


def main():
    ap = argparse.ArgumentParser(description="Vision attention: fused kernel vs SDPA + separate ops")
    ap.add_argument("-H", "--heads", type=int, default=16)
    ap.add_argument("-D", "--head-dim", type=int, default=72)
    ap.add_argument("--seq-len", type=int, default=64, help="patches per image")
    ap.add_argument("--num-seqs", type=int, default=4, help="number of images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cuda-graph", action="store_true", help="also capture + replay each path")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(args.seed)
    dev, dt = "cuda", torch.bfloat16
    H, D, seq_len, num_seqs = args.heads, args.head_dim, args.seq_len, args.num_seqs
    M = seq_len * num_seqs
    scaling = D**-0.5

    fva = _load_module(_KERNEL_SRC, "fused_vision_attention")
    fva.SPLIT_M_THRESHOLD = 0  # force DIRECT path (original split path drifts)

    qkv = torch.randn(M, 3 * H * D, device=dev, dtype=dt).mul_(0.5).contiguous()
    theta = torch.randn(M, D, device=dev) * 0.7
    cos = torch.cos(theta).to(torch.float32).contiguous()
    rsin = _signed_sin(torch.sin(theta).to(torch.float32), D)
    cu = torch.arange(0, (num_seqs + 1) * seq_len, seq_len, device=dev, dtype=torch.int32)

    def fused():
        return fva.forward(qkv, cos, rsin, cu, scaling, H, D)  # (M, H*D)

    def sdpa():
        return sdpa_path(qkv, cos, rsin, scaling, H, D, num_seqs)

    t_fused = _time(fused, args.warmup, args.iters)
    t_sdpa = _time(sdpa, args.warmup, args.iters)

    ref = sdpa().float()
    out = fused().float()
    rel = (ref - out).abs().max().item() / ref.abs().max().clamp(min=1e-6).item()

    print("=" * 78)
    print(f"Vision attention  |  H={H} M={M} D={D} seq_len={seq_len} num_seqs={num_seqs}  bf16")
    print("=" * 78)
    print(f"  Fused Triton kernel        : {t_fused * 1000:8.2f} us")
    print(f"  SDPA + separate RoPE ops   : {t_sdpa * 1000:8.2f} us")
    print(f"  speedup (sdpa / fused)     : {t_sdpa / t_fused:.2f}x")
    print(f"  rel error (fused vs sdpa)  : {rel:.3e}")

    if args.cuda_graph:
        g_f, out_f = _capture(fused)
        g_s, out_s = _capture(sdpa)
        tg_f = _time(g_f.replay, args.warmup, args.iters)
        tg_s = _time(g_s.replay, args.warmup, args.iters)
        g_f.replay()
        g_s.replay()
        torch.cuda.synchronize()
        rel_g = (
            (out_s.float() - out_f.float()).abs().max().item()
            / out_s.float().abs().max().clamp(min=1e-6).item()
        )
        print("-" * 78)
        print(f"  Fused   + CUDA graph       : {tg_f * 1000:8.2f} us  ({t_fused / tg_f:.2f}x vs eager)")
        print(f"  SDPA    + CUDA graph       : {tg_s * 1000:8.2f} us  ({t_sdpa / tg_s:.2f}x vs eager)")
        print(f"  speedup (sdpa / fused)     : {tg_s / tg_f:.2f}x")
        print(f"  rel error (fused vs sdpa)  : {rel_g:.3e}")
    print("=" * 78)


if __name__ == "__main__":
    main()
