import lightning as L
import torch
import torchmetrics
from torch import nn, optim
from transformers import CLIPModel, CLIPProcessor

from .backbones.backbone_factory import BackboneABC
from .model_factory import register_model


@register_model(name="clip_object_clf")
class CLIPObjectClassifier(L.LightningModule):
    """Zero-shot or fine-tuned CLIP image classifier.

    Computes cosine similarity between image features and pre-computed text
    features derived from ``class_names`` prompts.

    Args:
        model_name: HuggingFace identifier for the CLIP model
            (e.g. ``"openai/clip-vit-base-patch32"``).
        class_names: List of text prompts, one per class.
    """

    def __init__(self, model_name: str, class_names: list[str]) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.class_names = class_names
        self.num_classes = len(class_names)

        # Pre-compute and normalize text features
        text_inputs = self.processor(
            text=self.class_names, return_tensors="pt", padding=True
        )
        text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        self.register_buffer("text_features", text_features.detach())

    def forward(self, image):
        # Unnormalize ImageNet mean and std before passing to CLIP processor
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
        image_tensor = image * std + mean
        image_tensor = torch.clamp(image_tensor, 0, 1)

        inputs = self.processor(
            images=image_tensor, return_tensors="pt", padding=True, do_rescale=False
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Move pre-computed text features to the correct device
        text_features = self.text_features.to(self.device)

        # Cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["class_label"]
        logits = self.forward(x)
        acc = torchmetrics.functional.accuracy(
            logits, y, task="multiclass", num_classes=self.num_classes
        )
        self.log("val/acc", acc, on_epoch=True, batch_size=x.shape[0])

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["image"]
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["class_label"]

        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log(
            "train/acc",
            torchmetrics.functional.accuracy(
                logits, y, task="multiclass", num_classes=self.num_classes
            ),
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
        )
        return loss

    def configure_optimizers(self):
        # Higher weight decay for better regularization
        optimizer = optim.AdamW(
            self.parameters(),
            lr=1e-5,  # Lower learning rate for fine-tuning
            weight_decay=0.05,
            betas=(0.9, 0.999),
        )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=1e-5,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=25,
                final_div_factor=1000,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


@register_model(name="object_clf")
class ObjectClassifierModel(L.LightningModule):
    """Backbone + linear head image classifier.

    Args:
        backbone: A ``BackboneABC`` instance that produces spatial feature maps.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        backbone: BackboneABC,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.backbone = backbone
        clf_channels = backbone.out_channels
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(clf_channels, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.clf(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["class_label"]

        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train/loss", loss, on_step=True)
        self.log(
            "train/acc",
            torchmetrics.functional.accuracy(
                logits, y, task="multiclass", num_classes=self.num_classes
            ),
            on_epoch=True,
            batch_size=x.shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["class_label"]
        pred = self.forward(x)
        acc = torchmetrics.functional.accuracy(
            pred, y, task="multiclass", num_classes=self.num_classes
        )
        self.log("val/acc", acc, on_epoch=True, batch_size=x.shape[0])

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=0.05,
            betas=(0.9, 0.999),
        )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = {
            "scheduler": optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=1e-4,
                total_steps=total_steps,
                pct_start=0.1,  # 10% warmup
                anneal_strategy="cos",
                div_factor=25,  # Start from lr/25
                final_div_factor=1000,  # End at lr/1000
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
