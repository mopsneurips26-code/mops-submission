import torch
from torchvision.transforms import v2
from torchvision.tv_tensors import Image, Mask


class MopsImageTransforms:
    """Image augmentation pipeline for MOPS-IL training.

    Applies geometric (affine) and photometric (color jitter, blur) augmentations
    consistently to paired RGB images and affordance masks. Images are normalized
    to [0, 1] and optionally standardized with ImageNet statistics.

    Args:
        use_imagenet_stats: If True, apply ImageNet mean/std normalization to images.
    """

    def __init__(
        self,
        use_imagenet_stats: bool = True,
    ) -> None:
        # Define the transforms. Note that ColorJitter is photometric, so it only
        # affects the image by default, while RandomAffine affects both.
        self.transforms = v2.Compose(
            [
                # Step 1: Geometric Transforms (Affine seems missing but recommended)
                v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                # Step 2: Photometric Transforms (Image only)
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                # Step 3: Convert to final Tensor format
                v2.ToDtype(
                    {Image: torch.float32, Mask: torch.uint8}, scale=True
                ),  # Convert image to float and normalize to [0, 1]
            ]
        )
        if use_imagenet_stats:
            self.normalize = v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            self.normalize = v2.Identity()

    def __call__(self, item: dict) -> dict:
        keys_to_transform = []
        processed_keys = set()

        for key in item:
            # Fallback in case state keys are added in the future
            if "images" not in key:
                continue

            # Pair image with its affordance mask if available
            # Only process if it is image, therefore no "affordance" in key
            if "affordance" not in key:
                img_key = key
                mask_key = img_key.replace("_image", "_segmentation_affordance")
                if mask_key in item:
                    keys_to_transform.append((img_key, mask_key))
                    processed_keys.add(img_key)
                    processed_keys.add(mask_key)
                else:
                    keys_to_transform.append((img_key, None))
                    processed_keys.add(img_key)
            else:
                continue

        # Now apply transforms
        for img_key, mask_key in keys_to_transform:
            if img_key and mask_key:
                img = Image(item[img_key])
                mask = Mask(item[mask_key])

                # Apply transform to pair
                out_img, out_mask = self.transforms(img, mask)
                out_img = self.normalize(out_img)

                # Update item
                item[img_key] = out_img
                item[mask_key] = out_mask

            elif img_key:
                img = Image(item[img_key])
                out_img = self.transforms(img)
                out_img = self.normalize(out_img)
                item[img_key] = out_img

        return item
