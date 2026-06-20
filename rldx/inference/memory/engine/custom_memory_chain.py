"""CustomMemoryChain: fused TransformerMemory pipeline as a single nn.Module.

Mirrors VLM's CustomLLMChain pattern, adapted for Memory:
  - Fused QKV GEMM (cuBLAS)
  - mem::fused_attention (Triton: RoPE + block-causal SDPA)
  - mem::fused_epilogue_add2_rmsnorm (Triton: residual add + RMSNorm)
  - SwiGLU MLP (cuBLAS + torch.compile handles fusion)

Cross-layer fusion: layer N's post-MLP residual add is fused with
layer N+1's input RMSNorm via a single Triton epilogue kernel.

RoPE: cos/signed_sin pre-computed at chain init from block-wise position IDs.
  Matches eager: cos/sin computed fp32 → cast to bf16. Multiply in bf16.
"""

from __future__ import annotations

import memory.engine.ops  # noqa: F401 — registers mem:: ops
import torch
import torch.nn as nn
import torch.nn.functional as F

from rldx.utils.dist import rank_zero_print as _print

from .kernels.fused_memory_attention import prepare_fused_qkv_weight, prepare_rope_buffers


class MemoryLayerParam(nn.Module):
    """Per-layer weight container for TransformerDecoderLayer (no forward)."""

    def __init__(self, layer):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm

        self.register_buffer("qkv_weight", prepare_fused_qkv_weight(layer))
        self.register_buffer("o_proj_weight", layer.self_attn.o_proj.weight.data)
        self.register_buffer("gate_proj_weight", layer.mlp.gate_proj.weight.data)
        self.register_buffer("up_proj_weight", layer.mlp.up_proj.weight.data)
        self.register_buffer("down_proj_weight", layer.mlp.down_proj.weight.data)


class CustomMemoryChain(nn.Module):
    """Chain of fused TransformerDecoderLayers for TransformerMemory.

    Uses custom Triton ops for attention and cross-layer epilogue fusion.
    cuBLAS GEMMs (via F.linear) for projections and MLP.
    torch.compile optimizes and fuses remaining PyTorch ops.

    Args:
        gs_memory: GraphSafeMemory instance
        device: target device
        dtype: compute dtype (bf16)
    """

    def __init__(self, gs_memory, device=None, dtype=torch.bfloat16):
        super().__init__()
        memory = gs_memory._memory

        self.num_heads = memory.config.num_attention_heads
        self.head_dim = memory.hidden_size // self.num_heads
        self.block_attn_size = memory.block_attn_size
        # mem::fused_attention only implements BLOCK-causal. The eager model uses
        # a STANDARD causal mask when use_causal_attn=True (block_attn_size only
        # gates the non-causal block-attention path). Standard causal ==
        # block-causal with block=1, so pass 1 to the kernel in that case.
        self.attn_block_size = 1 if memory.use_causal_attn else memory.block_attn_size

        self.layers = nn.ModuleList([MemoryLayerParam(layer) for layer in memory.layers])
        self.norm = memory.norm
        self.n_layers = len(memory.layers)

        # Pre-compute RoPE buffers (shared across all layers)
        if memory.use_rope:
            rotary_emb = memory.layers[0].self_attn.rotary_emb
            position_ids = gs_memory.static_position_ids  # (1, M)
            dev = device or next(memory.parameters()).device
            cos, signed_sin = prepare_rope_buffers(rotary_emb, position_ids, dev, dtype)
            self.register_buffer("cos", cos)
            self.register_buffer("signed_sin", signed_sin)

        # Attention backend: False = fused Triton (RoPE + block-causal),
        # True = eager baked-RoPE + F.sdpa. Toggle on the instance (e.g. bench Path F).
        self._use_sdpa = False

    def _sdpa_attention(self, qkv, num_heads, head_dim):
        """SDPA path: eager baked-RoPE + F.scaled_dot_product_attention (block-causal).

        Numerically matches ``mem::fused_attention``: ``signed_sin`` already has
        the rotate_half sign folded in, so RoPE is ``q*cos + swap_half(q)*signed_sin``
        (swap the two head halves, no negation). Mask is block-causal — position i
        attends j iff ``i // block >= j // block`` (block == 1 → standard causal).
        scale = head_dim**-0.5 (matches the kernel's 1/sqrt(D)).
        """
        M = qkv.shape[0]
        QD = num_heads * head_dim
        half = head_dim // 2
        scaling = head_dim ** -0.5

        q = qkv[:, :QD].reshape(M, num_heads, head_dim).float()
        k = qkv[:, QD : 2 * QD].reshape(M, num_heads, head_dim).float()
        v = qkv[:, 2 * QD : 3 * QD].reshape(M, num_heads, head_dim)

        cos = self.cos.view(M, 1, head_dim)
        sin = self.signed_sin.view(M, 1, head_dim)
        q_sw = torch.cat((q[..., half:], q[..., :half]), dim=-1)
        k_sw = torch.cat((k[..., half:], k[..., :half]), dim=-1)
        q = (q * cos + q_sw * sin).to(v.dtype)
        k = (k * cos + k_sw * sin).to(v.dtype)

        # (M, H, D) -> (1, H, M, D): one block-causal sequence
        q = q.reshape(1, M, num_heads, head_dim).transpose(1, 2)
        k = k.reshape(1, M, num_heads, head_dim).transpose(1, 2)
        v = v.reshape(1, M, num_heads, head_dim).transpose(1, 2)

        blk = self.attn_block_size
        if blk == 1:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scaling)
        else:
            idx = torch.arange(M, device=qkv.device)
            mask = (idx[:, None] // blk) >= (idx[None, :] // blk)  # (M, M) bool, True = attend
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scaling)
        return out.transpose(1, 2).reshape(M, QD)  # (M, Q_DIM)

    def forward(self, inputs_embeds):
        """Forward pass with fused ops.

        Args:
            inputs_embeds: (B, M, D)

        Returns:
            (B, M, D)
        """
        B, M, D = inputs_embeds.shape
        hidden_states = inputs_embeds
        cos = self.cos
        ssin = self.signed_sin
        n_layers = self.n_layers

        # First layer's input LayerNorm (no previous epilogue to fuse with)
        normed = self.layers[0].input_layernorm(hidden_states)

        for i, layer in enumerate(self.layers):
            # 1. QKV projection — single fused GEMM (cuBLAS)
            qkv = F.linear(normed.view(M, D), layer.qkv_weight)  # (M, QKV_DIM)

            # 2. Attention — fused Triton (RoPE + block-causal) OR eager RoPE + F.sdpa
            if self._use_sdpa:
                attn_out = self._sdpa_attention(qkv, self.num_heads, self.head_dim)
            else:
                attn_out = torch.ops.mem.fused_attention(
                    qkv,
                    cos,
                    ssin,
                    self.num_heads,
                    self.head_dim,
                    self.attn_block_size,
                )  # (M, Q_DIM)

            # 3. O projection (cuBLAS)
            attn_out = F.linear(attn_out, layer.o_proj_weight).view(B, M, -1)

            # 4. Post-attention epilogue — Triton (residual add + post_attn LayerNorm)
            hidden_states, post_attn_normed = torch.ops.mem.fused_epilogue_add2_rmsnorm(
                attn_out, hidden_states, layer.post_attention_layernorm.weight
            )

            # 5. MLP: SwiGLU (cuBLAS GEMMs + torch.compile handles SiLU fusion)
            gate = F.linear(post_attn_normed, layer.gate_proj_weight)
            up = F.linear(post_attn_normed, layer.up_proj_weight)
            mlp_out = F.linear(F.silu(gate) * up, layer.down_proj_weight)

            # 6. Post-MLP epilogue
            if i < n_layers - 1:
                # Cross-layer fusion: residual add + next layer's input LayerNorm
                next_norm_weight = self.layers[i + 1].input_layernorm.weight
                hidden_states, normed = torch.ops.mem.fused_epilogue_add2_rmsnorm(
                    mlp_out, hidden_states, next_norm_weight
                )
            else:
                # Last layer: plain residual add
                hidden_states = hidden_states + mlp_out

        # Final norm
        return self.norm(hidden_states)


def build_custom_memory_chain(gs_memory, device=None, dtype=torch.bfloat16):
    """Build a CustomMemoryChain from a GraphSafeMemory (no compilation).

    Args:
        gs_memory: GraphSafeMemory instance

    Returns:
        CustomMemoryChain (uncompiled)
    """
    return CustomMemoryChain(gs_memory, device=device, dtype=dtype).eval()


def compile_custom_memory_chain(chain, sample_input, compile_mode="max-autotune", fullgraph=True):
    """Compile a CustomMemoryChain with torch.compile and trigger compilation.

    Args:
        chain: CustomMemoryChain instance
        sample_input: (B, S, D) tensor for warmup
        compile_mode: torch.compile mode
        fullgraph: when True (default) torch.compile errors on any
            graph break, forcing the whole forward into one FX graph.

    Returns:
        (compiled_chain, compile_time_s)
    """
    import time as _time

    _print(f"  [MemoryChain] Compiling ({compile_mode}, fullgraph={fullgraph})...")
    compiled = torch.compile(chain, mode=compile_mode, fullgraph=fullgraph)

    t0 = _time.time()
    with torch.no_grad():
        compiled(sample_input)
    torch.cuda.synchronize()
    compile_time = _time.time() - t0
    _print(f"  [MemoryChain] Compilation: {compile_time:.1f}s")

    return compiled, compile_time
