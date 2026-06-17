"""Model registry for the inference benchmarks.

Each entry pairs a checkpoint with the kwargs the loader passes through
to ``VTCQwen3VLBackbone``. The three production entries mirror the
RLDX-1 deployment lineup:

    rldx_1_pretrain         — base, video-only (no memory / physics)
    rldx_1_midtrain_allex   — video + motion + memory + torque
    rldx_1_midtrain_droid   — video + motion + memory + tactile + torque
"""

from __future__ import annotations

from pathlib import Path


# PT and MT-ALLEX ship their processor config under a ``processor/``
# subfolder, but the loader reads ``processor_config.json`` straight from
# the path root (no ``subfolder`` support). Point those entries at a
# locally-materialised processor dir instead. Populate it once with:
#   snapshot_download(repo, allow_patterns="processor/*",
#                     local_dir=_PROC_CACHE / repo.split("/")[-1])
# MT-DROID keeps its processor at the repo root, so it stays a HF path.
_PROC_CACHE = Path(__file__).resolve().parents[3] / ".proc_cache"


MODEL_REGISTRY = {
    "rldx_1_pretrain": {
        "hf_path": "RLWRLD/RLDX-1-PT",
        "processor_path": str(_PROC_CACHE / "RLDX-1-PT" / "processor"),
        # Backbone-only — the rest of the action-model plumbing is
        # initialised fresh by the loader.
        "load_mode": "extract_backbone",
        "default_args": {
            "use_cog_tokens": True,
            "n_cog_tokens": 64,
            "select_layer": 18,
        },
    },
    "rldx_1_midtrain_allex": {
        "hf_path": "RLWRLD/RLDX-1-MT-ALLEX",
        "processor_path": str(_PROC_CACHE / "RLDX-1-MT-ALLEX" / "processor"),
        # Carries memory + torque physics weights — load the full model.
        "load_mode": "full",
        "default_args": {
            "use_cog_tokens": True,
            "n_cog_tokens": 64,
            "select_layer": 18,
        },
    },
    "rldx_1_midtrain_droid": {
        "hf_path": "RLWRLD/RLDX-1-MT-DROID",
        "processor_path": "RLWRLD/RLDX-1-MT-DROID",
        # Same shape as ALLEX with tactile + torque physics.
        "load_mode": "full",
        "default_args": {
            "use_cog_tokens": True,
            "n_cog_tokens": 64,
            "select_layer": 18,
        },
    },
    # VTC backbone variant; loaded via the dynamic-import string.
    "vtc_qwen3_vl_8b": {
        "hf_path": "RLWRLD/RLDX-1-VLM",
        "backbone_cls": "rldx.model.modules.backbone.adapter.VTCQwen3VLBackbone",
    },
}
