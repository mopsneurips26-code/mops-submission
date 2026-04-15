# Getting Started — Dataset from Scratch

This guide walks through creating the MOPS-CaSa dataset from raw RoboCasa demonstrations.

> **Shortcut:** If you only want to train and evaluate using the public MOPS-CaSa dataset, see the [README](README.md).

---

## 1. Installation

```bash
uv sync && source .venv/bin/activate
```

---

## 2. Dataset Preparation

This section describes how to create the MOPS-CaSa dataset from scratch using
RoboCasa human demonstrations. **If you already have the dataset, skip to
[Training](#3-training).**

### 2.1 Download RoboCasa Assets

```bash
python src/robocasa/scripts/download_kitchen_assets.py
python src/robocasa/scripts/download_datasets.py --ds_types human_raw
```

### 2.2 Record Affordance Observations

Extract RGB images, affordance segmentation masks, and robot state from the raw
simulation states. This replays each demonstration trajectory in MuJoCo and renders
observations with semantic annotations.

**Single dataset:**

```bash
python src/mopscasa/scripts/record_affordances.py \
    --dataset <path/to/demo.hdf5> \
    --suffix mops
```

**Batch (all datasets under a directory):**

```bash
python src/mopscasa/scripts/multi_record_affordances.py \
    --root_dir <path/to/robocasa/datasets> \
    --suffix mops
```

### 2.3 Convert to LeRobot Format

Convert the recorded HDF5 files to the LeRobot dataset format used by the training
pipeline.

**Single dataset:**

```bash
python src/mopscasa/scripts/convert_lerobot.py \
    --datasets <path/to/recorded.hdf5> \
    --output_path data/mops-casa
```

**Batch conversion:**

```bash
python src/mopscasa/scripts/multi_convert_lerobot.py \
    --datasets <path/to/recorded_datasets> \
    --output_path data/mops-casa
```

The result is a LeRobot-compatible dataset under `data/`:

```
data/mops-casa/
├── meta/          # Dataset metadata, episode info, task descriptions
├── data/          # Parquet files with state, action, and index data
└── videos/        # MP4 video files (RGB + affordance per camera)
```

Once the dataset is ready, follow the training and evaluation instructions in the [README](README.md).
