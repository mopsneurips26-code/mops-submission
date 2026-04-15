import argparse

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from mops_pred.config import DatasetConfig
from mops_pred.datasets.dataset_factory import create_dataloader
from mops_pred.models.dinov2_segmentation import DINOv2SegmentationModel
from mops_pred.models.dinov3_segmentation import DINOv3SegmentationModel

MODEL_CLASSES = {
    "dinov2_segmentation": DINOv2SegmentationModel,
    "dinov3_segmentation": DINOv3SegmentationModel,
}


def visualize_segmentation(
    images, predictions, targets, num_samples: int = 4, output_path: str | None = None
):
    """Visualize segmentation predictions."""
    num_samples = min(num_samples, len(images))
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        axs = axs[np.newaxis, :]

    for i in range(num_samples):
        # Unnormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = images[i].cpu() * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()

        # Get prediction and target masks
        pred_mask = predictions[i].cpu().numpy()
        target_mask = targets[i].cpu().squeeze().numpy()

        # Plot
        axs[i, 0].imshow(img)
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(target_mask, cmap="tab20")
        axs[i, 1].set_title("Ground Truth")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(pred_mask, cmap="tab20")
        axs[i, 2].set_title("Prediction (Zero-Shot)")
        axs[i, 2].axis("off")

    plt.tight_layout()
    outfile = output_path or "dinov2_zeroshot_segmentation_results.png"
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Visualization saved to {outfile}")


def _get_dataloaders(cfg: dict):
    dataset_cfg = cfg["dataset"]
    return create_dataloader(
        DatasetConfig(
            name=dataset_cfg["name"],
            data_dir=dataset_cfg["data_dir"],
            labels=dataset_cfg["labels"],
        ),
        batch_size=cfg["training"]["batch_size"],
        augment=False,
    )


def run_experiment(cfg: dict):
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(42)

    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    task = model_cfg["task"]
    model_cls = MODEL_CLASSES[model_cfg["name"]]
    model_name_tag = model_cfg["model_name"].replace("/", "-")
    experiment_tag = f"{model_cfg['name']}-{model_name_tag}-zeroshot-{task}"

    model = model_cls(
        num_classes=model_cfg["num_classes"],
        task=task,
        model_name=model_cfg["model_name"],
        freeze_backbone=model_cfg.get("freeze_backbone", True),
        lr=model_cfg.get("lr", 1e-3),
        multilabel=model_cfg.get("multilabel", False),
        hidden_dim=model_cfg.get("hidden_dim", 256),
    )
    train_dl, test_dl = _get_dataloaders(cfg)

    print(f"\n{'=' * 60}")
    print(f"Zero-Shot Segmentation Testing :: {experiment_tag}")
    print(f"Task: {task}")
    print(f"Num Classes: {model_cfg['num_classes']}")
    print("Backbone: FROZEN (linear probing)")
    print(f"{'=' * 60}\n")

    wandb_project = cfg.get("wandb", {}).get("project", "mops-pred-2026")
    wandb_logger = WandbLogger(project=wandb_project, config=cfg, name=experiment_tag)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/iou",
        dirpath=wandb_logger.experiment.dir,
        filename="best",
        save_top_k=1,
        mode="max",
    )

    trainer = L.Trainer(
        max_epochs=training_cfg["num_epochs"],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    print("Training segmentation head (linear probing)...")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    print("Linear probing complete.")

    print("\nLoading best model and running final validation...")
    best_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    if best_path is None:
        raise RuntimeError("No checkpoint was saved during training.")

    best_model = model_cls.load_from_checkpoint(best_path)
    trainer.validate(best_model, dataloaders=test_dl)
    print("Final validation complete.")

    print("\nGenerating prediction visualizations...")
    best_model.eval()
    device = getattr(trainer.strategy, "root_device", best_model.device)
    best_model.to(device)
    for batch in test_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            predictions = best_model.predict_step(batch, 0)["predictions"]

        visualize_segmentation(
            batch["image"],
            predictions,
            batch[task],
            num_samples=min(4, len(batch["image"])),
            output_path=f"{experiment_tag}_results.png",
        )
        break

    print("\nZero-shot testing complete!")
    print(f"Best model saved to: {best_path}")
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="DINO zero-shot segmentation")
    parser.add_argument("--config_path", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
