"""Vision-encoder CHAIN benchmark — lettered paths (mirrors the LLM/memory/action benches).

Benchmarks the vision encoder in path variants, all producing the same merged output:

  A: Vanilla (GraphSafe eager)      — stock graph-safe vision encoder (gs_visual), eager.
                                      NO custom ops (standard PyTorch attention). Reference.
  C: GraphSafe + CUDA Graph         — gs_visual + manual CUDA-graph capture.
  D: Custom Chain (fused) + compile — CustomVisionEncoderChain, fused Triton attention
                                      (baked RoPE + non-causal varlen attn in one kernel) + torch.compile.
  E: Custom Chain (SDPA) + compile  — same chain but attention = eager baked-RoPE +
                                      F.scaled_dot_product_attention (cuDNN/FA) + torch.compile.
  F: GraphSafe + compile            — gs_visual + torch.compile, NO custom ops (all-PyTorch
                                      compiler-only counterpart to D).

D and E share the SAME chain (GEMMs / norms / MLP / epilogues); only the attention
sub-op differs (chain._use_sdpa). A / C / F use zero custom ops. Correctness is vs Path A.

Usage:
  python inference/backbone/vision_encoder/benchmark_vision_chain.py
  python inference/backbone/vision_encoder/benchmark_vision_chain.py --mode all --num-images 2
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

# Path setup: resolve `utils` (inference/utils) + `model`/`vision_encoder` (backbone/).
# NOTE: unlike benchmark_backbone.py we must NOT pin this file's dir (vision_encoder/),
# else `model`/`engine` would shadow the backbone-level packages. Pin backbone/ instead.
_INFERENCE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_BACKBONE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _INFERENCE)
import _path  # noqa: E402


_path.setup()  # adds PROJ_ROOT + inference/ (no caller pin)
sys.path.insert(0, _BACKBONE)  # pin backbone/ so model/vision_encoder resolve there

import backbone.vision_encoder.engine.ops  # noqa: E402,F401 — registers rldx_backbone:: vision ops
from vision_encoder.engine.custom_vision_encoder_chain import (  # noqa: E402
    CustomVisionEncoderChain,
)

from model import GraphSafeQwen3VLBackbone  # noqa: E402
from utils import (  # noqa: E402
    generate_synthetic_input,
    load_backbone,
    measure_times,
    print_correctness,
    print_latency_table,
)


_MODE_TO_MODEL_TYPE = {"video": "rldx_1_pretrain", "all": "rldx_1_midtrain_allex"}


def build_vision_chain(gs_visual):
    """Construct a CustomVisionEncoderChain from gs_visual (mirrors build_custom_backbone_chain)."""
    enable_motion_fast = os.environ.get("CUSTOM_VLM_ENABLE_MOTION_FAST") == "1"
    return CustomVisionEncoderChain(
        gs_visual.blocks,
        gs_visual.merger,
        gs_visual.deepstack_merger_list,
        gs_visual.deepstack_visual_indexes,
        gs_visual.pos_cos,
        gs_visual.pos_sin,
        gs_visual.cu_seqlens,
        gs_visual.max_seqlen,
        motion_block=gs_visual.motion_block,
        motion_insert_layer=gs_visual.motion_insert_layer,
        motion_grid_sizes=gs_visual.motion_grid_sizes,
        enable_motion_fast=enable_motion_fast,
    ).eval()


def main():
    ap = argparse.ArgumentParser(description="Vision encoder chain: fused Triton vs SDPA")
    ap.add_argument("--mode", default="video", choices=list(_MODE_TO_MODEL_TYPE.keys()))
    ap.add_argument("--num-images", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--concat-frames", action="store_true")
    ap.add_argument("--iter", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-compile", action="store_true", help="skip torch.compile paths")
    ap.add_argument(
        "--compile-mode", default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    ap.add_argument(
        "--inspect-dir", default=None,
        help="dump torch.compile codegen+kernels per backend here (via inductor_inspect), then exit",
    )
    args = ap.parse_args()
    args.model_type = _MODE_TO_MODEL_TYPE[args.mode]
    args.model_path = None

    backbone, meta = load_backbone(args)
    device = meta["device"]

    processor_path = meta["model_cfg"].get(
        "processor_path", meta["model_cfg"]["hf_path"]
    )
    vl_input, input_info = generate_synthetic_input(
        processor_path, args.num_images, args.image_size, args.image_size,
        args.concat_frames, device, args.seed, custom_prompt=None,
    )

    # GraphSafe wrap → gs_visual (vision encoder with static buffers)
    num_frames = input_info.get("num_images", 1) if args.concat_frames else 1
    gs = GraphSafeQwen3VLBackbone(backbone, vl_input, num_frames=num_frames, num_views=1)
    gs_visual = gs.gs_visual

    # Vision-chain input = patch_embed(pixel_values) + static pos embeds
    pv = vl_input["pixel_values"]
    if pv.ndim == 3:
        pv = pv.reshape(-1, pv.shape[-1])
    pv = pv.type(gs_visual.dtype)
    with torch.no_grad():
        hidden = gs_visual.patch_embed(pv) + gs_visual.pos_embeds
    hidden = hidden.contiguous()

    chain = build_vision_chain(gs_visual)
    M, Dv = hidden.shape
    print("=" * 80)
    print(f"Vision Encoder CHAIN benchmark  |  mode={args.mode}")
    print("=" * 80)
    print(f"  blocks={chain.n_layers}, tokens M={M}, D={Dv}, num_seqs={chain._num_seqs}, "
          f"images={args.num_images}x{args.image_size}")
    print(f"  iter={args.iter}, warmup={args.warmup}")

    results = {}
    outputs = {}
    build_times = {}

    def run_benchmark(label, fn):
        with torch.no_grad():
            out = fn()
        out = out.detach().clone()
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"  [!] {label}: NaN/Inf")
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        results[label] = measure_times(fn, args.iter)
        outputs[label] = out

    def chain_fn(mod):  # CustomVisionEncoderChain: takes patch-embedded `hidden`
        def fn():
            with torch.no_grad():
                merged, _ = mod(hidden)
                return merged
        return fn

    def gs_fn(mod):  # stock graph-safe encoder (NO custom ops): takes raw pixel_values
        def fn():
            with torch.no_grad():
                merged, _ = mod(pv)
                return merged
        return fn

    # =========================================================================
    # Path A: Vanilla (GraphSafe vision encoder, eager) — reference
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path A: Vanilla (GraphSafe vision encoder, eager)")
    print(f"{'=' * 60}")
    run_benchmark("A: Vanilla", gs_fn(gs_visual))

    # =========================================================================
    # Path C: GraphSafe + CUDA Graph
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("Path C: GraphSafe + CUDA Graph")
    print(f"{'=' * 60}")
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                gs_visual(pv)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            cg_merged, _ = gs_visual(pv)
        torch.cuda.synchronize()
        print("  CUDA graph captured")

        def cg_fn():
            g.replay()
            return cg_merged

        run_benchmark("C: GraphSafe + CG", cg_fn)
    except Exception as e:
        print(f"  [GraphSafe + CG] Failed: {e}")
        traceback.print_exc()

    # =========================================================================
    # Path D / E: Custom chain (fused / SDPA attention) + torch.compile
    # =========================================================================
    if not args.no_compile:
        for use_sdpa, label in [
            (False, "D: Custom(fused) + compile"),
            (True, "E: Custom(sdpa) + compile"),
        ]:
            try:
                chain._use_sdpa = use_sdpa
                torch._dynamo.reset()
                compiled = torch.compile(chain, mode=args.compile_mode)
                print(f"\n[compile] {label} (mode={args.compile_mode}) ...")
                t0 = _time.time()
                with torch.no_grad():
                    compiled(hidden)
                torch.cuda.synchronize()
                build_times[label] = _time.time() - t0
                run_benchmark(label, chain_fn(compiled))
            except Exception as e:
                print(f"  [{label}] failed: {e}")
                traceback.print_exc()
            finally:
                torch._dynamo.reset()

        # =====================================================================
        # Path F: GraphSafe + torch.compile (NO custom ops)
        # =====================================================================
        try:
            torch._dynamo.reset()
            compiled_gs = torch.compile(gs_visual, mode=args.compile_mode)
            print(f"\n[compile] F: GraphSafe + compile (mode={args.compile_mode}) ...")
            t0 = _time.time()
            with torch.no_grad():
                compiled_gs(pv)
            torch.cuda.synchronize()
            build_times["F: GraphSafe + compile"] = _time.time() - t0
            run_benchmark("F: GraphSafe + compile", gs_fn(compiled_gs))
        except Exception as e:
            print(f"  [F: GraphSafe + compile] failed: {e}")
            traceback.print_exc()
        finally:
            torch._dynamo.reset()

    print()
    print("=" * 80)
    print(f"Vision Encoder CHAIN — mode={args.mode}, M={M}, blocks={chain.n_layers}")
    print("=" * 80)
    if build_times:
        print("Build: " + ", ".join(f"{k}={v:.1f}s" for k, v in build_times.items()))
    print_latency_table("Full vision-encoder chain latency", results)

    if "A: Vanilla" in outputs and len(outputs) > 1:
        ref = outputs["A: Vanilla"]
        entries = [(lbl, o) for lbl, o in outputs.items() if lbl != "A: Vanilla"]
        print_correctness("Correctness (vs A: Vanilla)", entries, ref)

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated(device) / (1024**2):.1f} MB\n")


if __name__ == "__main__":
    main()
