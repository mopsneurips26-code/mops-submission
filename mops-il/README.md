# MOPS-IL — Affordance-Guided Imitation Learning

Imitation learning for robotic manipulation using flow-matching policies with auxiliary affordance segmentation. Extends action generation with object-level semantic grounding.

> **Note:** This project includes vendored forks of [LeRobot](https://github.com/huggingface/lerobot) and [RoboCasa](https://github.com/robocasa/robocasa) under `src/`. Changes are minimal — primarily to support lossless encoding of affordance masks.

## Quick Start

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync && source .venv/bin/activate
```

> **Torchcodec:** If you encounter CUDA errors, install manually:
> `pip install torchcodec --index-url=https://download.pytorch.org/whl/cu130`

### Dataset

Download the MOPS-CaSa dataset (LeRobot format) from Hugging Face and place it under `data/`:

```
data/mops-casa/
  meta/     # metadata, episode info, task descriptions
  data/     # parquet files (state, action, index)
  videos/   # MP4 files (RGB + affordance per camera)
```

To create a dataset from scratch using RoboCasa demonstrations, see [GETTING_STARTED.md](GETTING_STARTED.md).

### Training

```bash
# Train MopsFlow (full model with affordance segmentation)
python src/mops_il/scripts/train.py \
    --dataset_path data/mops-casa \
    --policy mopsflow \
    --steps 100000

# Multi-GPU with Accelerate
accelerate launch src/mops_il/scripts/train.py \
    --dataset_path data/mops-casa \
    --policy mopsflow --batch_size 32
```

**Policy variants:** `ditflow` (DiT + flow matching), `affflow` (+ affordance conditioning), `mopsflow` (+ auxiliary segmentation decoder), `diffusion` (1D U-Net baseline).

Logs to [Weights & Biases](https://wandb.ai) by default. Disable with `--disable_wandb`.

### Evaluation

```bash
python src/mops_il/scripts/evaluate.py \
    --checkpoint outputs/train/<run>/checkpoints/<step>/pretrained_model \
    --num_workers 4 --max_trials 10
```

### Validation (Action Visualization)

```bash
python src/mops_il/scripts/validate.py \
    --checkpoint outputs/train/<run> \
    --episodes 0 1 2 --output validation_output
```

## Project Structure

| Directory | Description |
|---|---|
| `src/mops_il/` | Training loop, configs, policies (DiTFlow, AffFlow, MopsFlow) |
| `src/mopscasa/` | RoboCasa integration, evaluation, dataset conversion |
| `src/lerobot/` | Vendored LeRobot fork |
| `src/robocasa/` | Vendored RoboCasa fork |
