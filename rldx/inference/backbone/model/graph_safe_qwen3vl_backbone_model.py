"""Unified graph-safe VLM: Vision + Glue + LLM + Projection.

Combines GraphSafeQwen3VLVisionModel and GraphSafeQwen3VLTextModel with the
embedding/scatter/cog-token glue into a single nn.Module. All data-dependent
operations are pre-computed in __init__; forward() is a pure computation graph.

Engine builders (TRT, CustomVLMChain) receive this model and decompose it via
the public gs_visual / gs_text attributes for engine-specific optimization.

The cog-token slot is appended after the language IDs and routed through
the LLM as additional tokens whose embeddings come from a learned
``backbone.cog_emb`` parameter.
"""

from __future__ import annotations

from llm.model.graph_safe_qwen3vl_text_model import GraphSafeQwen3VLTextModel
import torch
import torch.nn as nn
from vision_encoder.model.graph_safe_qwen3vl_vision_model import GraphSafeQwen3VLVisionModel

from rldx.utils.dist import rank_zero_print as _print


class GraphSafeQwen3VLBackbone(nn.Module):
    """Unified graph-safe VLM model.

    Forward flow:
      pixel_values → Vision Encoder → Embed + Image Scatter →
      (cog-token append) → LLM Decoder (with compression) →
      (cog-token extract) → Projection → backbone_features

    Public attributes for engine builders:
      gs_visual:    GraphSafeQwen3VLVisionModel (vision static buffers)
      gs_text:      GraphSafeQwen3VLTextModel   (LLM static buffers)
      embed_tokens: nn.Embedding
      qwen_linear:  nn.Module (projection)
    """

    def __init__(self, backbone, vl_input, num_frames=1, num_views=1, pad_token_id=None):
        super().__init__()

        inner_model = backbone.qwen_model.model  # Qwen3VLModel
        visual = inner_model.visual
        language_model = inner_model.language_model

        input_ids = vl_input["input_ids"]
        grid_thw = vl_input["image_grid_thw"]
        device = input_ids.device

        # --- Cog-token config ---
        self.n_cog_tokens = (
            getattr(backbone, "n_cog_tokens", 0)
            if getattr(backbone, "use_cog_tokens", False)
            else 0
        )
        self.cog_mode = getattr(backbone, "cog_mode", "cog_only")

        # --- Sub-models ---
        _print("\nSetting up GraphSafeQwen3VLBackbone...")
        self.gs_visual = GraphSafeQwen3VLVisionModel(
            visual, grid_thw, num_frames=num_frames, num_views=num_views
        )

        # Determine num_views for compression (matches vanilla LayerWrapper logic)
        iwe = vl_input.get("image_wise_encoding")
        if iwe is not None:
            if isinstance(iwe, torch.Tensor):
                iwe_val = bool(iwe.flatten()[0].item())
            else:
                iwe_val = bool(iwe)
        else:
            iwe_val = False
        compress_num_views = vl_input.get("num_views") if iwe_val else None

        self.gs_text = GraphSafeQwen3VLTextModel(
            language_model, input_ids, self.n_cog_tokens, num_views=compress_num_views
        )

        # Install into backbone so engines that still read from there find them
        inner_model.visual = self.gs_visual
        inner_model.language_model = self.gs_text

        # --- Shared modules (references, not owned) ---
        self.embed_tokens = self.gs_text._text_model.embed_tokens
        self.qwen_linear = backbone.qwen_linear
        self.image_token_id = inner_model.config.image_token_id
        # Pad id used by ``update_input_ids`` to refill a shorter instruction
        # up to the baked length. Falls back to Qwen ``<|endoftext|>`` (151643)
        # when the config carries no explicit pad token.
        _cfg_pad = getattr(inner_model.config, "pad_token_id", None)
        self.pad_token_id = (
            pad_token_id
            if pad_token_id is not None
            else (_cfg_pad if _cfg_pad is not None else 151643)
        )
        # Floor of the special-token id range (Qwen3: added/special tokens —
        # image_pad, vision_start/end, im_start/end, <|endoftext|> — are all
        # >= 151643; normal vocab is below). ``update_input_ids`` only lets
        # normal-vocab (text) token VALUES change at the baked positions.
        self.special_id_min = 151643

        # --- Static glue buffers ---
        B, L_ids = input_ids.shape
        D = self.embed_tokens.embedding_dim

        self.register_buffer("static_input_ids", input_ids.clone())

        image_mask = input_ids == self.image_token_id
        self.register_buffer("image_mask_3d", image_mask.unsqueeze(-1).expand(B, L_ids, D))

        # 3D MROPE position IDs (matches the cog-token append in the vanilla
        # backbone forward).
        # Qwen3VL uses different position IDs per axis (temporal/height/width)
        # for image tokens based on their spatial grid layout.
        L_full = L_ids + self.n_cog_tokens
        if self.n_cog_tokens > 0:
            placeholder_token_id = 248068
            meta_ids = torch.full(
                (B, self.n_cog_tokens), placeholder_token_id, dtype=input_ids.dtype, device=device
            )
            extended_input_ids = torch.cat([input_ids, meta_ids], dim=1)
        else:
            extended_input_ids = input_ids

        with torch.no_grad():
            position_ids, _ = inner_model.get_rope_index(extended_input_ids, grid_thw, None)
        # position_ids: (3, B, L_full) — proper 3D MROPE
        self.register_buffer("static_position_ids", position_ids)

        # Cog-token embedding parameter.
        if self.n_cog_tokens > 0 and hasattr(backbone, "cog_emb"):
            self.register_buffer("static_cog_emb", backbone.cog_emb.data.clone())
        else:
            self.static_cog_emb = None

        n_img = image_mask.sum().item()
        _print(
            f"  Unified backbone: B={B}, L_ids={L_ids}, D={D}, n_cog_tokens={self.n_cog_tokens}, "
            f"L_full={L_full}, image_tokens={n_img}"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, vl_input):
        """Full VLM forward: vl_input → backbone_features.

        Args:
            vl_input: dict with at least 'pixel_values' key.
                      Other keys (input_ids, attention_mask, etc.) are ignored
                      — the model uses its own pre-computed static buffers.

        Returns:
            backbone_features: (B, M_out, D_proj) tensor
        """
        # Vision encoding
        pixel_values = vl_input["pixel_values"]
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        pixel_values = pixel_values.type(self.gs_visual.dtype)

        image_emb, deepstack_features = self.gs_visual(hidden_states=pixel_values, grid_thw=None)

        # Token embedding + image scatter
        dtype = self.embed_tokens.weight.dtype
        image_emb = image_emb.to(dtype=dtype)

        token_emb = self.embed_tokens(self.static_input_ids)
        token_emb = token_emb.masked_scatter(self.image_mask_3d, image_emb)

        # cog-token append
        if self.n_cog_tokens > 0 and self.static_cog_emb is not None:
            meta = self.static_cog_emb.to(dtype).unsqueeze(0).expand(token_emb.size(0), -1, -1)
            full_emb = torch.cat([token_emb, meta], dim=1)
        else:
            full_emb = token_emb

        # DeepStack — scatter vision features into full sequence for LLM
        # Build visual_pos_mask for full_emb (L_ids + cog-token)
        # image_mask_3d covers L_ids; pad with False for cog-token positions
        deepstack_add = None
        if len(deepstack_features) > 0:
            B_ds = full_emb.shape[0]
            L_full_ds = full_emb.shape[1]
            D_ds = full_emb.shape[2]
            # image_mask_3d: (B, L_ids, D) → take 2D mask, pad to L_full
            vis_mask_2d = self.image_mask_3d[:, :, 0]  # (B, L_ids) bool
            vis_mask_full = torch.cat(
                [
                    vis_mask_2d,
                    torch.zeros(B_ds, self.n_cog_tokens, dtype=torch.bool, device=full_emb.device),
                ],
                dim=1,
            )  # (B, L_full)
            vis_mask_full_3d = vis_mask_full.unsqueeze(-1).expand(B_ds, L_full_ds, D_ds)

            ds_list = []
            for ds_feat in deepstack_features:
                ds_full = torch.zeros_like(full_emb)
                ds_full = ds_full.masked_scatter(vis_mask_full_3d, ds_feat.to(dtype))
                ds_list.append(ds_full)
            deepstack_add = torch.stack(ds_list, dim=0)  # (N_ds, B, L_full, D)

        # LLM forward
        lm_out = self.gs_text(
            inputs_embeds=full_emb,
            position_ids=self.static_position_ids,
            deepstack_add=deepstack_add,
        )
        hidden_states = lm_out.last_hidden_state

        # cog-token extract — must match the vanilla backbone slice at
        # ``rldx/model/modules/backbone/adapter.py`` (cog_mode='cog_only').
        # Without the slice GraphSafeVLM and CustomVLMChain disagree on
        # token count: the custom chain slices unconditionally on
        # ``n_cog_tokens > 0`` (``custom_backbone_chain.py``) — Path D
        # crashes at compile time and Path C silently mis-computes
        # (cos_sim ≈ 0.96 vs vanilla).
        if self.n_cog_tokens > 0 and self.cog_mode == "cog_only":
            hidden_states = hidden_states[:, -self.n_cog_tokens :, :]

        # Projection
        return self.qwen_linear(hidden_states)

    # ------------------------------------------------------------------
    # Runtime instruction refresh (graph-safe)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_input_ids(self, new_input_ids, pad_token_id=None, allow_approx_pad=False):
        """Refresh the baked instruction tokens for graph-safe reuse.

        Writes ``new_input_ids`` into ``static_input_ids`` IN PLACE so a
        captured CUDA graph / compiled kernel picks up the new instruction on
        the next replay — ``forward`` reads ``self.static_input_ids`` for the
        token-embedding gather, and an in-place ``copy_`` keeps the buffer
        address the graph captured. Without this the model keeps serving the
        instruction baked at build time.

        ``image_mask_3d``, ``static_position_ids`` and the compression
        ``begin/end`` indices are baked at build time and NOT recomputed here,
        so they stay valid only when the new sequence preserves the positions
        of every structural/image token. Two regimes:

          * **Same length (exact).** Only normal-vocab (text) token values may
            differ; every baked special token (image_pad, vision_start/end,
            im_start/end, ...) must stay at its position with the same id, and
            no new special may appear at a baked text position. Bit-exact vs
            vanilla — verified at runtime.
          * **Shorter (approximate, opt-in via ``allow_approx_pad``).** The
            instruction is right-padded to the baked length. This is NOT
            bit-exact: padded positions are attended under the mask-free varlen
            path and trailing structural tokens shift in MROPE. Disabled by
            default so the caller falls back to the exact eager path instead.

        Longer-than-baked instructions are always rejected.

        Returns:
            bool: True if applied (graph stays valid); False if the caller
                  should run the vanilla path instead.
        """
        B, L = self.static_input_ids.shape
        ids = new_input_ids
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(device=self.static_input_ids.device, dtype=self.static_input_ids.dtype)
        if ids.shape[0] != B:
            return False

        if ids.shape[1] == L:
            # Exact: structural/image tokens must match the baked layout, only
            # text token values may change.
            special = self.static_input_ids >= self.special_id_min
            if not torch.equal(ids[special], self.static_input_ids[special]):
                return False
            if bool((ids[~special] >= self.special_id_min).any()):
                return False
        elif ids.shape[1] < L and allow_approx_pad:
            pad = pad_token_id if pad_token_id is not None else self.pad_token_id
            ids = torch.cat([ids, ids.new_full((B, L - ids.shape[1]), pad)], dim=1)
            # Looser guard for the approximate path: image positions must align.
            if not torch.equal(ids == self.image_token_id, self.image_mask_3d[:, :, 0]):
                return False
        else:
            return False

        self.static_input_ids.copy_(ids)
        return True
