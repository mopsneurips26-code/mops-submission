# MOPS-Data — Synthetic Dataset Generation

Generates photoreal synthetic datasets (RGB, depth, segmentation masks, surface normals) by rendering [PartNet-Mobility](https://sapien.ucsd.edu/browse) objects in [ManiSkill 3](https://github.com/haosulab/ManiSkill) / SAPIEN simulations. Output is in WebDataset format (sharded TARs), ready for Hugging Face Hub publishing.

## Quick Start

**Requires Python 3.10** (ManiSkill 3 constraint) and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .
```

### Asset Setup

1. **PartNet-Mobility** (required for all datasets): download from [SAPIEN](https://sapien.ucsd.edu/browse) and place under `data/partnet_mobility/`.
2. **RoboCasa assets** (required for kitchen dataset only):
   ```bash
   python -m mani_skill.utils.download_asset RoboCasa
   ```

### Generate Datasets

Each script has a `--debug` flag for a fast sanity check with small images and few samples.

```bash
# Single-object: isolated objects, multiple viewpoints, varied lighting
python scripts/generate_single_object.py --debug

# Kitchen: objects in RoboCasa kitchens, table-level + overhead views
python scripts/generate_kitchen.py --debug

# Clutter: cluttered tabletop scenes, top-down views
python scripts/generate_clutter.py --debug
```

Drop `--debug` for full production runs. Use `--output <path>` and `--format hdf5` to customize output.

## Project Structure

| Directory | Description |
|---|---|
| `scripts/` | Dataset generation entry points |
| `demos/` | Visualization scripts |
| `src/mops_data/generation/` | Pipeline configs and writers (WebDataset / HDF5) |
| `src/mops_data/envs/` | Custom ManiSkill 3 environments |
| `src/mops_data/asset_manager/` | PartNet-Mobility annotation loading |
| `src/mops_data/render/` | Observation augmentation and shaders |

