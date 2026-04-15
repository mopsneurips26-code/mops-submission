# MOPS-Pred — Vision Experiments

Object classification and affordance/semantic segmentation on MOPS synthetic datasets. Uses PyTorch Lightning for training and Weights & Biases for logging.

## Quick Start

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync && source .venv/bin/activate
```

### Dataset

Download the datasets from Hugging Face and place them under `data/mops_data/`.

### Training

All experiments are config-driven via YAML files under `configs/`.

```bash
# Debug run (fast_dev_run, no WandB upload)
python -m mops_pred.training --config_path configs/deeplabv3_affordance_kitchen.yaml --mode debug

# Full training
python -m mops_pred.training --config_path configs/deeplabv3_affordance_kitchen.yaml --mode train

# Resume from checkpoint
python -m mops_pred.training --config_path configs/deeplabv3_affordance_kitchen.yaml --checkpoint path/to/ckpt.pt
```

Any config field can be overridden from the CLI, e.g. `--training.batch_size 16`.

### Available Configs

| Config | Task |
|---|---|
| `deeplabv3_affordance_kitchen.yaml` | Affordance segmentation (kitchen scenes) |
| `deeplabv3_affordance_clutter.yaml` | Affordance segmentation (clutter scenes) |
| `deeplabv3_semantic.yaml` | Semantic segmentation |
| `segformer_affordance_kitchen.yaml` | SegFormer affordance segmentation |
| `dinov3_affordance_kitchen.yaml` | DINOv3 affordance segmentation |
| `resnet50_clf.yaml` | ResNet-50 object classification |
| `vit_clf.yaml` | ViT object classification |


## Project Structure

| Directory | Description |
|---|---|
| `configs/` | YAML experiment configs |
| `mops_pred/training.py` | Main training entry point |
| `mops_pred/config.py` | Config dataclasses (draccus) |
| `mops_pred/models/` | Lightning modules + backbone/model registries |
| `mops_pred/datasets/` | Dataset implementations + dataset registry |
