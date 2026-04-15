import lightning as L
import torch
import torchmetrics
from torch import nn, optim
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

from .model_factory import register_model
from .segm_transformer import FocalLoss


@register_model(name="segmentation")
class SegmentationModel(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        task: str = "semantic",
        lr: float = 1e-4,
        multilabel: bool = False,
        loss: str = "bce",  # Add loss hyperparameter
    ) -> None:
        """
        Initializes a DeepLabV3 model for semantic or multilabel segmentation.

        Args:
            num_classes: Number of segmentation classes.
            task: The key for the target mask in the batch (e.g., 'semantic').
            lr: Learning rate for the optimizer.
            multilabel: If True, performs multilabel segmentation; if False, semantic segmentation.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        in_channels = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits from the model."""
        return self.model(x)["out"]

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
