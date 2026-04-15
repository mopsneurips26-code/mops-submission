from typing import List

import torch
import torchvision.transforms.v2 as T_v2
from torchvision.transforms.functional import to_tensor

from mops_pred.datasets.base_h5 import BaseH5Dataset
from mops_pred.datasets.dataset_factory import register_dataset


@register_dataset(name="object_centric")
class ObjectCentricDataset(BaseH5Dataset):
    """Dataset for object-centric classification tasks."""

    def __init__(
        self,
        h5_path: str,
        train: bool = True,
        labels: List[str] | None = None,
        augment: bool = False,
    ):
        super().__init__(h5_path, train, augment)
        self.labels = labels if labels is not None else ["class"]

        # Define transforms using the modern v2 API
        if self.augment:
            self.transform = T_v2.Compose(
                [
                    T_v2.RandomResizedCrop(
                        size=(224, 224), scale=(0.8, 1.0), antialias=True
                    ),
                    T_v2.RandomHorizontalFlip(p=0.5),
                    T_v2.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                    ),
                    T_v2.RandomRotation(degrees=10),
                ]
            )
        else:
            self.transform = T_v2.Resize(size=(224, 224), antialias=True)

        self.normalize = T_v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, idx):
        base_sample = super().__getitem__(idx)
        actual_idx = base_sample["actual_idx"]

        # Load image, convert to tensor, and apply transforms
        image_np = self.h5_file["images"][base_sample["image_id"]][:]
        image = to_tensor(image_np)
        image = self.transform(image)

        sample = {"image": self.normalize(image), "image_id": base_sample["image_id"]}

        # Load class label
        if "class" in self.labels:
            class_idx = self.h5_file["labels"]["class_labels"][actual_idx]
            sample["class_label"] = torch.tensor(class_idx, dtype=torch.long)

        return sample
