#!/bin/bash
#SBATCH --job-name="Finetune RLDX-1 on custom dataset"
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --output=slurm_out/%j-rldx1_ft_custom.out
#SBATCH --error=slurm_out/%j-rldx1_ft_custom.err

set -euo pipefail

# Custom-dataset fine-tune template — vanilla video VLA on your LeRobot
# dataset(s). Toggle motion / memory / physics by uncommenting lines in the
# ARGS array below. Edit dataset paths, modality config, and step counts
# in-place. See run_scripts/train/examples/README.md for details.

export WANDB_PROJECT="${WANDB_PROJECT:-rldx-finetune}"
export NO_ALBUMENTATIONS_UPDATE=1

BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CKPT_NAME="rldx1_ft_custom"
RUN_NAME="$CKPT_NAME"

NUM_GPUS=8
BASE_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
EMBODIMENT_TAG="GENERAL_EMBODIMENT"  # or OXE_FRACTAL / OXE_BRIDGE_ORIG / ...
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/your_dataset_config.py"  # see rldx/configs/data/ for examples
CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/$CKPT_NAME"
COLOR_JITTER_PARAMS="brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"

ARGS=(
    --n-cog-tokens 64
    --video-length 4

    # Dataset (uncomment the multi-dataset line to mix datasets instead;
    # --dataset-paths takes precedence over --dataset-path at runtime)
    --dataset-path "/path/to/your/lerobot_dataset"
    # --dataset-paths "/path/a" "/path/b" --dataset-mix-ratios 1.0 0.5

    --dataloader-num-workers 8
    --embodiment-tag "$EMBODIMENT_TAG"
    --modality-config-path "$MODALITY_CONFIG_PATH"
    --color-jitter-params $COLOR_JITTER_PARAMS
    --base-model-path "$BASE_MODEL_PATH"
    --output-dir "$CKPT_DIR"
    --num-gpus "$NUM_GPUS"
    --save-total-limit 5
    --save-steps 1000
    --max-steps 60000
    --global-batch-size 256

    # Add-ons (uncomment any to enable)

    # Motion: discrete action-prefix tokens injected mid-backbone.
    # --use-motion --motion-insert-layer 9

    # Memory: cross-clip context tokens for long-horizon manipulation.
    # --use-memory --memory-length 4 --memory-stride 16 --memory-n-cog-tokens 16 --concat-memory

    # Physics: per-step force/torque head; match keys/dims to your schema.
    # --use-physics --physics-keys torque --physics-dims 48 --allow-missing-physics

    --use-wandb
    --wandb-project "$WANDB_PROJECT"
    --experiment-name "$RUN_NAME"
)

cd "$BASE_DIR"
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    rldx/experiment/launch_train.py "${ARGS[@]}"
