# Getting Started with RLDX-1

A guided tour of the RLDX-1 codebase via five notebooks. Read them in
order the first time — each one assumes the previous.

| # | Notebook | Covers |
|---|---|---|
| 1 | [`01_quickstart.ipynb`](01_quickstart.ipynb) | Install check, load a pre-trained model, run a single inference step on a dummy observation. |
| 2 | [`02_dataset_preparation.ipynb`](02_dataset_preparation.ipynb) | LeRobot dataset layout, modality configs, embodiment registration, what the processor expects. |
| 3 | [`03_finetuning.ipynb`](03_finetuning.ipynb) | `launch_train.py` CLI, core flags, checkpoint layout, a one-step validation run. |
| 4 | [`04_inference_server.ipynb`](04_inference_server.ipynb) | Starting the inference server, connecting a client, and the sim-policy wrapper. |
| 5 | [`05_module_guide.ipynb`](05_module_guide.ipynb) | Opt-in modules: video, memory module, motion module, physics, cognition tokens. Flags, data requirements, composition matrix. |

## Prerequisites

- A working install per [`docs/installation.md`](../docs/installation.md):
  ```bash
  uv sync --python 3.10 && uv pip install -e .
  ```
- A CUDA-capable GPU for notebooks 1, 3, and 4. Notebooks 2 and 5 are CPU-only.
- ~30 GB of free disk for the pre-trained-checkpoint cache.
- Optional: a LeRobot-format dataset on local disk for the training /
  dataset notebooks.

## Running the notebooks

Jupyter is not declared as a dev dependency of `rldx`. Install it into
the existing `uv` environment:

```bash
uv pip install jupyterlab
uv run jupyter lab getting_started/
```

Or open individual notebooks in VS Code / Cursor and select
`.venv/bin/python` as the interpreter.

## Default checkpoints

The notebooks load `RLWRLD/RLDX-1-PT` as the fine-tune / mid-train base,
and `RLWRLD/RLDX-1-FT-ROBOCASA` for the inference quickstart. Override
the `MODEL` / `BASE_MODEL` constant at the top of each notebook to load
a different HuggingFace repo or a local checkpoint directory.
