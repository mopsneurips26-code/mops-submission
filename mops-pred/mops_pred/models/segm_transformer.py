import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn, optim
from transformers import SegformerForSemanticSegmentation

from .model_factory import register_model


class MaskedMultilabelJaccardIndex(torchmetrics.Metric):
    """Multilabel Jaccard Index (IoU) computed only on masked pixels."""

    def __init__(self, num_labels: int, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.add_state(
            "intersection", default=torch.zeros(num_labels), dist_reduce_fx="sum"
        )
        self.add_state("union", default=torch.zeros(num_labels), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        # preds: logits (B, C, H, W), target: (B, C, H, W), mask: (B, 1, H, W) or (B, H, W)
        preds_binary = torch.sigmoid(preds) > 0.5
        target_bool = target.bool()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = mask.bool().expand_as(preds_binary)
        self.intersection += (preds_binary & target_bool & mask).sum(dim=(0, 2, 3))
        self.union += ((preds_binary | target_bool) & mask).sum(dim=(0, 2, 3))

    def compute(self):
        iou_per_class = self.intersection / self.union.clamp(min=1)
        return iou_per_class.mean()


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        p = torch.sigmoid(inputs)
        pt = p * targets + (1 - p) * (1 - targets)
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


@register_model(name="segformer")
class TransformerSegmentationModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_name: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        task: str = "semantic",
        lr: float = 1e-4,
        multilabel: bool = False,
        loss: str = "bce",  # Add loss hyperparameter
        partnet_iou: bool = False,
    ) -> None:
        """
        Initializes a SegFormer model for semantic or multilabel segmentation.

        Args:
            num_classes: Number of segmentation classes.
            model_name: Name of the pretrained SegFormer model from Hugging Face.
            task: The key for the target mask in the batch (e.g., 'semantic').
            lr: Learning rate for the optimizer.
            multilabel: If True, performs multilabel segmentation; if False, semantic segmentation.
            loss: The loss function to use for multilabel tasks ('bce' or 'focal').
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            self.hparams.model_name,
            num_labels=self.hparams.num_classes,
            ignore_mismatched_sizes=True,  # Replaces the head with a new one
        )

        # Define loss and metrics based on task type
        if self.hparams.multilabel:
            if self.hparams.loss == "focal":
                self.loss_fn = FocalLoss()
            else:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits from the model, upsampled to the input size."""
        # Get the raw logits from the SegFormer model
        logits = self.model(pixel_values=x).logits
        # Upsample logits to match the input image size
        upsampled_logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return upsampled_logits

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
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)

        # We need the total number of training steps for OneCycleLR
        # trainer.estimated_stepping_batches is available after the trainer is initialized
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
                    "interval": "step",  # Call scheduler on every step
                },
            }
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for inference."""
        x = batch["image"]
        logits = self.forward(x)

        if self.hparams.multilabel:
            probs = torch.sigmoid(logits)
            # Threshold at 0.5 for binary predictions per class
            preds = (probs > 0.5).int()
            return {"probabilities": probs, "predictions": preds}
        else:
            # Get the class with the highest logit value
            preds = torch.argmax(logits, dim=1)
            return {"predictions": preds, "logits": logits}
