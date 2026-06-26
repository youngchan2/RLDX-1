# RLDX Training Scripts

The default `BASE_MODEL_PATH` for fine-tune scripts is
`RLWRLD/RLDX-1-PT`; override with `BASE_MODEL_PATH=<path>` to start from a
different base.

## Directory layout

| Directory | What's inside |
|---|---|
| [`benchmarks/`](benchmarks/) | One script per benchmark — the recipes that produced each released `RLWRLD/RLDX-1-FT-*` checkpoint (LIBERO, GR-1, RoboCasa, RoboCasa365, SIMPLER). |
| [`examples/`](examples/) | Training recipes that produce `RLWRLD/RLDX-1-MT-*`, plus a [`finetune.sh`](examples/finetune.sh) template to copy + edit for fine-tuning on your own custom dataset. |

## Per-benchmark fine-tune

The benchmark guides under [`run_scripts/eval/<bench>/README.md`](../eval/)
each link to the matching script in `benchmarks/`. As a quick
reference:

| Benchmark | Script |
|---|---|
| LIBERO | `benchmarks/finetune_rldx1_libero.sh` |
| GR-1 Tabletop | `benchmarks/finetune_rldx1_gr1.sh` |
| RoboCasa Kitchen | `benchmarks/finetune_rldx1_robocasa.sh` |
| RoboCasa365 | `benchmarks/finetune_rldx1_robocasa365.sh` |
| SIMPLER Google | `benchmarks/finetune_rldx1_simpler_google.sh` |
| SIMPLER WidowX | `benchmarks/finetune_rldx1_simpler_widowx.sh` |

## Fine-tune on your own dataset

Copy [`examples/finetune.sh`](examples/finetune.sh), edit dataset paths
and modality config to match your data, and toggle motion / memory /
physics add-ons by uncommenting lines in the `ARGS=(...)` block. See
[`examples/README.md`](examples/) for the full add-on flag reference and
multi-dataset usage.
