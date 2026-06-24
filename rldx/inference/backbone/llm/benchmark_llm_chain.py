"""
LLM decoder CHAIN benchmark — isolated Qwen3VL text decoder stack.

Benchmarks the LLM decoder stack (N decoder layers + final RMSNorm) in the same
path family as the other module benchmarks, on a fixed synthetic hidden-state
input. Every path runs the SAME layer weights; only the attention/op backend and
acceleration differ:

  A: Vanilla (GraphSafe eager)   — graph-safe text model, eager. Faithful to the
                                   original text model (same math, only data-dependent
                                   graph-breaks removed) — used as the correctness reference.
  C: GraphSafe + CUDA Graph      — graph-safe text model + manual CUDA-graph capture.
  D: CustomLLMChain + compile    — custom Triton fused attention (QK-RMSNorm + RoPE +
                                   causal) + fused residual/RMSNorm epilogues, + torch.compile.
  E: CustomLLMChain(SDPA)+compile— SAME chain/epilogues as D, but attention runs as
                                   eager baked-RoPE + F.scaled_dot_product_attention
                                   (enable_gqa=True) instead of the fused Triton kernel.
                                   Isolates the attention-backend cost (D vs E).
  F: GraphSafe + compile         — graph-safe text model + torch.compile, NO custom ops
                                   (the all-PyTorch / compiler-only counterpart to D).

Notes:
  - No "B: Inductor on raw vanilla": the graph-safe eager model IS the faithful
    vanilla here (running the raw HF text model standalone needs hand-assembled
    position_embeddings + FA varlen kwargs).
  - Paths D & E share one CustomLLMChain instance (same weights); only the
    ``_use_sdpa`` toggle differs, so their latency delta is purely attention backend.
  - Input is synthetic randn(B, L, D); all paths receive the SAME tensor, so the
    latency and cross-path cos-sim are both meaningful. L / D / position_ids come
    from the real GraphSafe backbone (3D MROPE), not invented.

Usage:
  python inference/backbone/llm/benchmark_llm_chain.py
  python inference/backbone/llm/benchmark_llm_chain.py --mode all
"""

import argparse
import ctypes
import os
import sys
import time as _time
import traceback


# Fix NVRTC builtins path (mirrors benchmark_backbone.py)
try:
    import nvidia.cu13 as _cu13

    _cu13_lib = os.path.join(os.path.dirname(os.path.abspath(_cu13.__path__[0])), "cu13", "lib")
    if os.path.isdir(_cu13_lib):
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if _cu13_lib not in _ld:
            os.environ["LD_LIBRARY_PATH"] = f"{_cu13_lib}:{_ld}" if _ld else _cu13_lib
        _builtins = os.path.join(_cu13_lib, "libnvrtc-builtins.so.13.0")
        if os.path.isfile(_builtins):
            ctypes.CDLL(_builtins)
except (ImportError, OSError):
    pass

import torch
import torch._inductor.config as _inductor_config

from transformers.modeling_utils import str_to_torch_dtype


if "C64" not in str_to_torch_dtype:
    str_to_torch_dtype["C64"] = torch.complex64

_inductor_config.max_autotune_gemm_backends = "ATEN"
_inductor_config.emulate_precision_casts = True
_inductor_config.triton.cudagraph_trees = False  # needed for Path C manual CUDA-graph capture

# Path setup: this file lives in backbone/llm/. Add inference/ (for utils + the
# backbone.* package) and pin backbone/ so `llm.*` / `model` resolve as
# backbone-level packages (mirrors benchmark_vision_chain.py at the same depth).
_INFERENCE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_BACKBONE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _INFERENCE)
import _path  # noqa: E402

_path.setup()
sys.path.insert(0, _BACKBONE)

import backbone.llm.engine.ops  # noqa: E402,F401 — registers rldx_backbone:: fused epilogue ops
from llm.engine.custom_llm_chain import CustomLLMChain  # noqa: E402
from llm.engine.kernels.fused_llm_attention import prepare_signed_sin  # noqa: E402

from model import GraphSafeQwen3VLBackbone  # noqa: E402
from utils import (  # noqa: E402
    generate_synthetic_input,
    load_backbone,
    measure_times,
    print_correctness,
    print_latency_table,
)


_MODE_TO_MODEL_TYPE = {"video": "rldx_1_pretrain", "all": "rldx_1_midtrain_allex"}


def main():
    ap = argparse.ArgumentParser(description="Isolated LLM decoder chain benchmark")
    ap.add_argument("--mode", default="video", choices=list(_MODE_TO_MODEL_TYPE.keys()))
    ap.add_argument("--num-images", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--concat-frames", action="store_true")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--iter", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--compile-mode", default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    args = ap.parse_args()
    args.model_type = _MODE_TO_MODEL_TYPE[args.mode]
    args.model_path = None

    torch.cuda.set_device(args.device)
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    backbone, meta = load_backbone(args)
    device = meta["device"]

    processor_path = meta["model_cfg"].get("processor_path", meta["model_cfg"]["hf_path"])
    vl_input, input_info = generate_synthetic_input(
        processor_path, args.num_images, args.image_size, args.image_size,
        args.concat_frames, device, args.seed, custom_prompt=args.prompt,
    )

    # GraphSafe backbone → gs_text (text decoder) + static 3D-MROPE position ids.
    num_frames = input_info.get("num_images", 1) if args.concat_frames else 1
    gs_backbone = GraphSafeQwen3VLBackbone(backbone, vl_input, num_frames=num_frames, num_views=1)
    gs_text = gs_backbone.gs_text
    position_ids = gs_backbone.static_position_ids  # (3, B, L) MROPE

    if gs_text.compress_info is not None:
        raise SystemExit(
            "This isolated LLM bench assumes no VTC compression (single decoder stack). "
            f"compress_info={gs_text.compress_info}. Use a non-VTC model_type."
        )

    B, L = position_ids.shape[-2], position_ids.shape[-1]
    D = gs_backbone.embed_tokens.embedding_dim

    # Synthetic LLM input — same tensor for every path (fair latency + cos-sim).
    inputs_embeds = (torch.randn((B, L, D), device=device, dtype=dtype) * 0.02).contiguous()

    # ---- head dims (from first decoder layer; unwrap VTC LayerWrapper) ----
    first_raw = list(gs_text.layers)[0]
    if hasattr(first_raw, "layer") and hasattr(first_raw, "internal_projection"):
        first_raw = first_raw.layer
    head_dim = first_raw.self_attn.head_dim
    num_heads = first_raw.self_attn.q_proj.weight.shape[0] // head_dim
    num_kv_heads = first_raw.self_attn.k_proj.weight.shape[0] // head_dim
    n_layers = len(list(gs_text.layers))

    print("=" * 80)
    print(f"LLM decoder CHAIN benchmark  |  mode={args.mode} ({args.model_type})")
    print("=" * 80)
    print(f"  layers={n_layers}, tokens L={L}, D={D}, head_dim={head_dim}, "
          f"H_q={num_heads}, H_kv={num_kv_heads}")
    print(f"  iter={args.iter}, warmup={args.warmup}, compile-mode={args.compile_mode}")

    results = {}
    outputs = {}
    build_times = {}

    def run_benchmark(label, fn):
        with torch.no_grad():
            output = fn()
        output = output.detach().clone()
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"  [!] {label}: NaN/Inf detected")
        print(f"  Warming up ({args.warmup} iters)...")
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        print(f"  Benchmarking ({args.iter} iters)...")
        results[label] = measure_times(fn, args.iter)
        outputs[label] = output

    def gs_text_fn(mod):
        def fn():
            with torch.no_grad():
                out = mod(inputs_embeds=inputs_embeds, position_ids=position_ids, deepstack_add=None)
                return out.last_hidden_state
        return fn

    # =========================================================================
    # Path A: Vanilla (GraphSafe text model, eager) — reference
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path A: Vanilla (GraphSafe text model, eager)")
    print(f"{'=' * 60}")
    run_benchmark("A: Vanilla", gs_text_fn(gs_text))

    # =========================================================================
    # Path C: GraphSafe + CUDA Graph (manual capture)
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path C: GraphSafe + CUDA Graph")
    print(f"{'=' * 60}")
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                gs_text(inputs_embeds=inputs_embeds, position_ids=position_ids, deepstack_add=None)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            cg_out = gs_text(
                inputs_embeds=inputs_embeds, position_ids=position_ids, deepstack_add=None
            )
        torch.cuda.synchronize()
        print("  CUDA graph captured")

        def cg_fn():
            g.replay()
            return cg_out.last_hidden_state

        run_benchmark("C: GraphSafe + CG", cg_fn)
    except Exception as e:
        print(f"  [GraphSafe + CG] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Shared build for Paths D & E: RoPE buffers + one CustomLLMChain instance.
    # Both paths run identical weights/epilogues; only the ``_use_sdpa`` toggle
    # (fused Triton kernel vs F.sdpa) differs, so their latency delta is purely
    # the attention backend. Building once avoids duplicating chain weights.
    # =========================================================================
    llm_chain = None
    try:
        dummy_embeds = torch.empty(B, L, D, device=device, dtype=dtype)
        with torch.no_grad():
            pos_cos, pos_sin = gs_text.rotary_emb(dummy_embeds, position_ids)
        if pos_cos.dim() == 3:
            pos_cos = pos_cos.squeeze(0)
        if pos_sin.dim() == 3:
            pos_sin = pos_sin.squeeze(0)
        pos_cos = pos_cos.contiguous()
        signed_sin = prepare_signed_sin(pos_sin.contiguous(), head_dim)

        print("\n  Building CustomLLMChain (shared by Paths D & E)...")
        llm_chain = CustomLLMChain(list(gs_text.layers), gs_text.norm, pos_cos, signed_sin).eval()
    except Exception as e:
        print(f"  [CustomLLMChain build] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Path D: CustomLLMChain (fused Triton attention) + torch.compile
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path D: CustomLLMChain (fused Triton) + torch.compile")
    print(f"{'=' * 60}")
    if llm_chain is not None:
        try:
            llm_chain._use_sdpa = False  # fused Triton attention
            torch._dynamo.reset()
            compiled_chain = torch.compile(llm_chain, mode=args.compile_mode)

            def custom_fn():
                with torch.no_grad():
                    return compiled_chain(inputs_embeds)

            print(f"  Compiling (mode={args.compile_mode})...")
            t0 = _time.time()
            with torch.no_grad():
                compiled_chain(inputs_embeds)
            torch.cuda.synchronize()
            build_times["D: CustomLLMChain"] = _time.time() - t0
            print(f"  Compilation: {build_times['D: CustomLLMChain']:.1f}s")

            run_benchmark("D: CustomLLMChain", custom_fn)
            torch._dynamo.reset()
        except Exception as e:
            print(f"  [CustomLLMChain] Failed: {e}")
            traceback.print_exc()

    # =========================================================================
    # Path E: CustomLLMChain (SDPA attention, enable_gqa) + torch.compile
    # =========================================================================
    # Same chain instance as D with ``_use_sdpa=True``: attention becomes eager
    # baked-RoPE + F.scaled_dot_product_attention(enable_gqa=True). A fresh
    # torch.compile re-traces against the toggled branch (dynamo guards on the flag).
    print(f"\n{'=' * 60}")
    print("Path E: CustomLLMChain (SDPA, enable_gqa) + torch.compile")
    print(f"{'=' * 60}")
    if llm_chain is not None:
        try:
            llm_chain._use_sdpa = True  # eager RoPE + F.sdpa (GQA broadcast)
            torch._dynamo.reset()
            compiled_chain_sdpa = torch.compile(llm_chain, mode=args.compile_mode)

            def sdpa_fn():
                with torch.no_grad():
                    return compiled_chain_sdpa(inputs_embeds)

            print(f"  Compiling (mode={args.compile_mode})...")
            t0 = _time.time()
            with torch.no_grad():
                compiled_chain_sdpa(inputs_embeds)
            torch.cuda.synchronize()
            build_times["E: CustomLLMChain(SDPA)"] = _time.time() - t0
            print(f"  Compilation: {build_times['E: CustomLLMChain(SDPA)']:.1f}s")

            run_benchmark("E: CustomLLMChain(SDPA)", sdpa_fn)
            torch._dynamo.reset()
            llm_chain._use_sdpa = False  # restore default
        except Exception as e:
            print(f"  [CustomLLMChain SDPA] Failed: {e}")
            traceback.print_exc()

    # =========================================================================
    # Path F: GraphSafe + torch.compile (NO custom ops)
    # =========================================================================
    # Same graph-safe text model as Path A/C, but accelerated with torch.compile
    # instead of CUDA graph — the all-PyTorch counterpart to the custom chain (D).
    # Reusing the gs_text instance is safe: the manual CUDA-graph capture in Path C
    # records ops into a graph object and does not mutate the module.
    print(f"\n{'=' * 60}")
    print("Path F: GraphSafe + torch.compile (no custom ops)")
    print(f"{'=' * 60}")
    try:
        gs_text_compile = gs_backbone.gs_text  # torch.compile returns a fresh wrapper
        torch._dynamo.reset()
        compiled_gs = torch.compile(gs_text_compile, mode=args.compile_mode)

        print(f"  Compiling (mode={args.compile_mode})...")
        t0 = _time.time()
        with torch.no_grad():
            gs_text_fn(compiled_gs)()
        torch.cuda.synchronize()
        build_times["F: GraphSafe+compile"] = _time.time() - t0
        print(f"  Compilation: {build_times['F: GraphSafe+compile']:.1f}s")

        run_benchmark("F: GraphSafe + compile", gs_text_fn(compiled_gs))
        torch._dynamo.reset()
    except Exception as e:
        print(f"  [GraphSafe + compile] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Report
    # =========================================================================
    print()
    print("=" * 80)
    print(f"LLM decoder CHAIN benchmark — {args.model_type}")
    print("=" * 80)
    print(f"Config: layers={n_layers}, L={L}, D={D}, head_dim={head_dim}, "
          f"H_q={num_heads}, H_kv={num_kv_heads}")
    print(f"Iter={args.iter}, warmup={args.warmup}")
    if build_times:
        print(f"Build: {', '.join(f'{k}={v:.1f}s' for k, v in build_times.items())}")

    print_latency_table("LLM decoder stack latency", results)

    if "A: Vanilla" in outputs and len(outputs) > 1:
        ref = outputs["A: Vanilla"]
        corr_entries = [(label, out) for label, out in outputs.items() if label != "A: Vanilla"]
        print_correctness("Correctness (vs A: Vanilla)", corr_entries, ref)

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated(device) / (1024**2):.1f} MB\n")


if __name__ == "__main__":
    main()
