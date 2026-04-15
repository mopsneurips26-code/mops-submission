import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn, optim
from transformers import AutoModel

from .model_factory import register_model
from .segm_transformer import MaskedMultilabelJaccardIndex


class DINOv3SegmentationHead(nn.Module):
    """Segmentation head for DINOv3 features."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.decoder(x)


@register_model(name="dinov3_segmentation")
class DINOv3SegmentationModel(L.LightningModule):
    """Lightning module for DINOv3 zero-shot or linear-probe segmentation."""

    def __init__(
        self,
        num_classes: int,
        task: str = "semantic",
        lr: float = 1e-4,
        multilabel: bool = False,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze_backbone: bool = False,
        hidden_dim: int = 256,
        partnet_iou: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.backbone = AutoModel.from_pretrained(model_name)

        # Infer patch size and feature dimensions from the transformer config
        feature_dim = getattr(self.backbone.config, "hidden_size", None)
        if feature_dim is None:
            raise ValueError(
                "Backbone config missing hidden_size; ViT backbone expected."
            )

        patch_size = getattr(self.backbone.config, "patch_size", None)
        if patch_size is None:
            raise ValueError(
                "Backbone config missing patch_size; ViT backbone expected."
            )
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]

        # Get model's native image size for computing grid dimensions
        image_size = getattr(self.backbone.config, "image_size", None)
        if image_size is None:
            # DINOv3-ViT-B defaults to 518×518
            image_size = 518
        if isinstance(image_size, (list, tuple)):
            image_size_h, image_size_w = image_size
        else:
            image_size_h = image_size_w = image_size

        self.feature_dim = int(feature_dim)
        self.patch_size = int(patch_size)
        self.image_size_h = image_size_h
        self.image_size_w = image_size_w

        # Get number of register tokens (DINOv3 specific)
        self.num_register_tokens = getattr(
            self.backbone.config, "num_register_tokens", 0
        )

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.seg_head = DINOv3SegmentationHead(
            in_channels=self.feature_dim, num_classes=num_classes, hidden_dim=hidden_dim
        )

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

        if self.hparams.partnet_iou and self.hparams.multilabel:
            self.val_partnet_iou = MaskedMultilabelJaccardIndex(num_labels=num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch-grid features from the DINOv3 ViT backbone.

        DINOv3 outputs: [CLS, patch_tokens..., register_tokens...]
        We remove CLS and register tokens, keeping only patch tokens.
        """
        bsz, _, height, width = x.shape
        with torch.set_grad_enabled(not self.hparams.freeze_backbone):
            outputs = self.backbone(pixel_values=x)
            # Shape: (batch_size, num_tokens_with_cls_and_registers, hidden_size)
            all_tokens = outputs.last_hidden_state
            # Remove CLS token (first token) and register tokens (last N tokens)
            if self.num_register_tokens > 0:
                patch_tokens = all_tokens[:, 1 : -self.num_register_tokens, :]
            else:
                patch_tokens = all_tokens[:, 1:, :]

        # Compute grid dimensions based on actual input resolution
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        expected_patches = num_patches_h * num_patches_w
        actual_patches = patch_tokens.shape[1]

        if actual_patches != expected_patches:
            raise RuntimeError(
                f"Patch count mismatch: expected {expected_patches} "
                f"({num_patches_h}×{num_patches_w} from {height}×{width}) "
                f"but got {actual_patches} from backbone. "
                f"patch_size={self.patch_size}, register_tokens={self.num_register_tokens}. "
                f"Full token shape: {all_tokens.shape}"
            )

        # Reshape from (B, H*W, D) to (B, D, H, W)
        patch_tokens = patch_tokens.reshape(
            bsz, num_patches_h, num_patches_w, self.feature_dim
        ).permute(0, 3, 1, 2)
        return patch_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.seg_head(features)
        return F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

    def _common_step(self, batch, batch_idx, stage: str):
        x = batch["image"]
        y = batch[self.hparams.task]
        logits = self.forward(x)

        if self.hparams.multilabel:
            loss = self.loss_fn(logits, y.float())
            metrics = self.train_metrics if stage == "train" else self.val_metrics
            metrics.update(logits, y.int())
        else:
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
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self._common_step(batch, batch_idx, "val")
        if self.hparams.partnet_iou and "is_partnet" in batch:
            self.val_partnet_iou.update(
                logits, batch[self.hparams.task].int(), batch["is_partnet"]
            )
            self.log(
                "val/partnet_iou", self.val_partnet_iou, on_step=False, on_epoch=True
            )
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, lr=self.hparams.lr)

        if self.trainer:
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["image"]
        logits = self.forward(x)

        if self.hparams.multilabel:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            return {"probabilities": probs, "predictions": preds}

        preds = torch.argmax(logits, dim=1)
        return {"predictions": preds, "logits": logits}
