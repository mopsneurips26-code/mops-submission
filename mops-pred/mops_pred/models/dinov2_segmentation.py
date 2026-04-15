import lightning as L
import torch
import torchmetrics
from torch import nn, optim
import torch.nn.functional as F

from .model_factory import register_model


class DINOv2SegmentationHead(nn.Module):
    """Segmentation head for DINOv2 features."""

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


@register_model(name="dinov2_segmentation")
class DINOv2SegmentationModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        task: str = "semantic",
        lr: float = 1e-4,
        multilabel: bool = False,
        model_name: str = "dinov2_vits14",
        freeze_backbone: bool = False,
        hidden_dim: int = 256,
    ) -> None:
        """
        Initializes a DINOv2-based model for semantic or multilabel segmentation.

        Args:
            num_classes: Number of segmentation classes.
            task: The key for the target mask in the batch (e.g., 'semantic').
            lr: Learning rate for the optimizer.
            multilabel: If True, performs multilabel segmentation; if False, semantic segmentation.
            model_name: DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14').
            freeze_backbone: If True, freezes the DINOv2 backbone (for zero-shot or linear probing).
            hidden_dim: Hidden dimension for the segmentation head.
        """
        super().__init__()
        self.save_hyperparameters()

        # Load DINOv2 backbone
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

        # Get feature dimension from backbone
        if "vits14" in model_name:
            feature_dim = 384
            self.patch_size = 14
        elif "vitb14" in model_name:
            feature_dim = 768
            self.patch_size = 14
        elif "vitl14" in model_name:
            feature_dim = 1024
            self.patch_size = 14
        elif "vitg14" in model_name:
            feature_dim = 1536
            self.patch_size = 14
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Segmentation head
        self.seg_head = DINOv2SegmentationHead(
            in_channels=feature_dim, num_classes=num_classes, hidden_dim=hidden_dim
        )

        # Define loss and metrics based on task type
        if self.hparams.multilabel:
            self.loss_fn = nn.BCEWithLogitsLoss()
            metrics = torchmetrics.MetricCollection(
                {
                    "iou": torchmetrics.classification.MultilabelJaccardIndex(
                        num_labels=num_classes, average="macro"
                    ),
                    "f1": torchmetrics.classification.MultilabelF1Score(
                        num_labels=num_classes, average="macro"
                    ),
                }
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss()
            metrics = torchmetrics.MetricCollection(
                {
                    "iou": torchmetrics.classification.MulticlassJaccardIndex(
                        num_classes=num_classes, average="macro"
                    )
                }
            )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from DINOv2 backbone."""
        B, C, H, W = x.shape

        # Get patch embeddings from DINOv2
        with torch.set_grad_enabled(not self.hparams.freeze_backbone):
            features = self.backbone.forward_features(x)
            # features["x_norm_patchtokens"] has shape [B, num_patches, feature_dim]
            patch_features = features["x_norm_patchtokens"]

        # Reshape patch features to 2D feature map
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        patch_features = patch_features.reshape(
            B, num_patches_h, num_patches_w, -1
        ).permute(0, 3, 1, 2)

        return patch_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits from the model."""
        features = self.extract_features(x)
        logits = self.seg_head(features)

        # Upsample to input resolution
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return logits

    def _common_step(self, batch, batch_idx, stage: str):
        x = batch["image"]
        y = batch[self.hparams.task]
        logits = self.forward(x)

        if self.hparams.multilabel:
            # Target for BCE loss should be float
            loss = self.loss_fn(logits, y.float())
            # Metrics expect integer targets
            metrics = self.train_metrics if stage == "train" else self.val_metrics
            metrics.update(logits, y.int())
        else:
            # Target for cross-entropy should be long and squeezed
            loss = self.loss_fn(logits, y.long().squeeze(1))
            metrics = self.train_metrics if stage == "train" else self.val_metrics
            metrics.update(logits, y.long().squeeze(1))

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.hparams.lr)

        # We need the total number of training steps for OneCycleLR
        if self.trainer:
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for inference."""
        x = batch["image"]
        logits = self.forward(x)

        if self.hparams.multilabel:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            return {"probabilities": probs, "predictions": preds}
        else:
            preds = torch.argmax(logits, dim=1)
            return {"predictions": preds, "logits": logits}
