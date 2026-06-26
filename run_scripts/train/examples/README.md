# Training examples

Reference scripts that build on `RLWRLD/RLDX-1-PT`:

| Script | Purpose |
|---|---|
| [`finetune.sh`](finetune.sh) | Custom-dataset fine-tune template — vanilla video VLA on your LeRobot dataset(s). Toggle motion / memory / physics by uncommenting lines in the `ARGS` array. Supports both single-dataset (`--dataset-path`) and multi-dataset (`--dataset-paths` + `--dataset-mix-ratios`). |
| [`midtrain_rldx1_droid.sh`](midtrain_rldx1_droid.sh) | Mid-train on the DROID dataset to produce `RLWRLD/RLDX-1-MT-DROID` (memory + motion + physics enabled). |
| [`midtrain_rldx1_allex.sh`](midtrain_rldx1_allex.sh) | Mid-train on ALLEX humanoid data to produce `RLWRLD/RLDX-1-MT-ALLEX` (memory + motion + physics enabled). |

The default `BASE_MODEL_PATH` for all three is `RLWRLD/RLDX-1-PT`;
override with `BASE_MODEL_PATH=<path>` to start from a different base.

## Add-on CLI flags

The same flag set is used across `finetune.sh` (commented out by
default — uncomment to enable) and the mid-train scripts (always on):

```bash
# Memory: cross-clip context tokens for long-horizon manipulation
--use-memory --memory-length 4 --memory-stride 16 --memory-n-cog-tokens 16 --concat-memory

# Motion: discrete action-prefix tokens injected mid-backbone
--use-motion --motion-insert-layer 9

# Physics: per-step force/torque head; --physics-keys / --physics-dims
# must be aligned with your dataset's physics schema
--use-physics --physics-keys torque --physics-dims 48 --allow-missing-physics

# Recommended whenever any add-on above is enabled — warms up the
# newly-initialized weights for the first N steps
--new-param-warmup-steps 2000
```

## Multi-dataset training (finetune.sh)

`finetune.sh` accepts either a single LeRobot dataset root via
`--dataset-path`, or multiple roots via `--dataset-paths` (with optional
`--dataset-mix-ratios` for sampling weights). `--dataset-paths` takes
precedence over `--dataset-path` at runtime, so you can leave the
single-dataset line in place and just uncomment the multi-dataset line
below it.

The mid-train scripts use a different dataset spec
(`--pt-dataset-root` + `--pt-dataset-mix`, where the mix name is a
preset registered in `rldx/configs/data/`).

## Embodiment & modality config

Update `EMBODIMENT_TAG` and `MODALITY_CONFIG_PATH` in `finetune.sh` to
match your dataset. Built-in tags include `GENERAL_EMBODIMENT`,
`OXE_FRACTAL`, `OXE_BRIDGE_ORIG`, etc.; copy / edit one of the configs
under [`rldx/configs/data/`](../../../rldx/configs/data/) as a starting
point for the modality config.
