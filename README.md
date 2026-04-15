# MOPS — Multi-Objective Photoreal Simulation

This monorepo contains the code for **MOPS**, submitted to NeurIPS 2026.
Each subfolder is an independent project with its own environment and dependencies.

## Repository Structure

| Subfolder | Description | Python | Quick Link |
|---|---|---|---|
| [`mops-data`](mops-data/) | Synthetic dataset generation via ManiSkill 3 / SAPIEN | 3.10 | [README](mops-data/README.md) |
| [`mops-pred`](mops-pred/) | Computer vision experiments (classification, segmentation) | ≥ 3.11 | [README](mops-pred/README.md) |
| [`mops-il`](mops-il/) | Imitation learning on RoboCasa with affordance-augmented observations | ≥ 3.11 | [README](mops-il/README.md) |

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — used by all subprojects for environment and dependency management.
- **GPU** — a CUDA-capable GPU is expected for training and rendering.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

Each subproject is self-contained. `cd` into the relevant folder, create a virtual environment, and install:

```bash
# Example: mops-pred (vision experiments)
cd mops-pred
uv sync            # creates .venv and installs all deps
source .venv/bin/activate
```

See individual READMEs for dataset download instructions and runnable commands.

### mops-data — Dataset Generation

Generates photoreal synthetic datasets (RGB, depth, segmentation, normals) by rendering PartNet-Mobility objects in ManiSkill 3. Requires access to PartNet-Mobility assets (not bundled due to license restrictions).

```bash
cd mops-data
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .
python scripts/generate_single_object.py --debug   # quick sanity check
```

### mops-pred — Vision Experiments

Object classification and affordance segmentation experiments. Datasets are available on Hugging Face.

```bash
cd mops-pred
uv sync && source .venv/bin/activate
python -m mops_pred.training --config_path configs/deeplabv3_affordance_kitchen.yaml --mode debug
```

### mops-il — Imitation Learning

Flow-matching policies with auxiliary affordance segmentation for robotic manipulation. Includes vendored forks of [LeRobot](https://github.com/huggingface/lerobot) and [RoboCasa](https://github.com/robocasa/robocasa) with minimal changes to support lossless affordance mask encoding. Demonstration datasets are available in LeRobot format on Hugging Face.

```bash
cd mops-il
uv sync && source .venv/bin/activate
python src/mops_il/scripts/train.py \
    --dataset_path data/mops-casa \
    --policy mopsflow --steps 100000
```