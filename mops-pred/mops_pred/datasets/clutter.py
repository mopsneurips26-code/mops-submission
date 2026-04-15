from typing import List

import torch
import torchvision.transforms.v2 as T_v2
from torchvision import tv_tensors
from torchvision.transforms.functional import to_tensor

from mops_pred.datasets.base_h5 import BaseH5Dataset
from mops_pred.datasets.dataset_factory import register_dataset


@register_dataset(name="clutter")
class ClutterDataset(BaseH5Dataset):
    """Dataset for segmentation tasks on cluttered scenes."""

    def __init__(
        self,
        h5_path: str,
        train: bool = True,
        labels: List[str] | None = None,
        augment: bool = False,
    ):
        super().__init__(h5_path, train, augment)
        self.labels = labels if labels is not None else ["semantic"]

        # Define transforms
        if self.augment:
            self.spatial_transform = T_v2.Compose(
                [
                    T_v2.RandomResizedCrop(
                        size=(224, 224), scale=(0.8, 1.0), antialias=True
                    ),
                    T_v2.RandomHorizontalFlip(p=0.5),
                    T_v2.RandomRotation(degrees=10),
                ]
            )
            self.color_transform = T_v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            )
        else:
            self.spatial_transform = T_v2.Resize(size=(224, 224), antialias=True)
            self.color_transform = None

        self.normalize = T_v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __getitem__(self, idx):
        base_sample = super().__getitem__(idx)

        # Load image and masks
        image_np = self.h5_file["images"][base_sample["image_id"]][:]
        image = to_tensor(image_np)

        masks = {}
        for label_type in self.labels:
            if label_type == "class":
                continue  # Not a mask
            mask_data = self.h5_file["masks"][label_type][base_sample["image_id"]][:]
            # Ensure mask is a tensor with a channel dimension (C, H, W)
            if mask_data.ndim == 2:
                masks[label_type] = torch.from_numpy(mask_data).unsqueeze(0)
            else:  # Assumes H, W, C
                masks[label_type] = torch.from_numpy(mask_data).permute(2, 0, 1)

        # Load is_partnet metadata mask if available
        if "is_partnet" in self.h5_file["masks"]:
            pn = self.h5_file["masks"]["is_partnet"][base_sample["image_id"]][:]
            masks["is_partnet"] = (
                torch.from_numpy(pn).unsqueeze(0)
                if pn.ndim == 2
                else torch.from_numpy(pn).permute(2, 0, 1)
            )

        # Wrap tensors for synchronized transformations
        image = tv_tensors.Image(image)
        masks = {k: tv_tensors.Mask(v, dtype=v.dtype) for k, v in masks.items()}

        # Apply transforms
        image, masks = self.spatial_transform(image, masks)
        if self.color_transform:
            image = self.color_transform(image)

        sample = {"image": self.normalize(image), "image_id": base_sample["image_id"]}

        # Unwrap masks and add to sample, squeezing channel dim for segmentation targets
        for k, v in masks.items():
            sample[k] = v
        return sample
