import lightning as L
import torch

from mops_pred.config import DatasetConfig
from mops_pred.datasets.dataset_factory import create_dataloader
from mops_pred.models.object_classifier import CLIPObjectClassifier

# Hardcoded constants for testing
MODEL_NAME = "openai/clip-vit-base-patch32"
CLASS_NAMES = [
    "a rendering of a coffee machine",
    "a rendering of a switch",
    "a rendering of a USB thumb drive",
    "a rendering of a safe",
    "a rendering of a washing machine",
    "a rendering of a dispenser",
    "a rendering of a lighter",
    "a rendering of a globe",
    "a rendering of a window",
    "a rendering of a box",
    "a rendering of a computer mouse",
    "a rendering of a lamp",
    "a rendering of a chair",
    "a rendering of a trash can",
    "a rendering of eyeglasses",
    "a rendering of storage furniture",
    "a rendering of a faucet",
    "a rendering of a display screen",
    "a rendering of scissors",
    "a rendering of a pen",
    "a rendering of pliers",
    "a rendering of a kitchen pot",
    "a rendering of a fan",
    "a rendering of a laptop",
    "a rendering of a knife",
    "a rendering of a printer",
    "a rendering of a table",
    "a rendering of a suitcase",
    "a rendering of a kettle",
    "a rendering of a bucket",
    "a rendering of a refrigerator",
    "a rendering of a door",
    "a rendering of a bottle",
    "a rendering of a keyboard",
    "a rendering of a toaster",
    "a rendering of a clock",
    "a rendering of a dishwasher",
    "a rendering of a remote control",
    "a rendering of a camera",
    "a rendering of a toilet",
    "a rendering of a folding chair",
    "a rendering of a cart",
    "a rendering of a stapler",
    "a rendering of a phone",
    "a rendering of an oven",
    "a rendering of a microwave",
]
BATCH_SIZE = 4
IMAGE_SIZE = 224  # Standard for this CLIP model
NUM_CLASSES = 46


def test_clip_classifier():
    """
    Initializes and tests the CLIPObjectClassifier on a dummy dataset.
    """
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(42)

    # Initialize the model
    model = CLIPObjectClassifier(model_name=MODEL_NAME, class_names=CLASS_NAMES)

    train_dl, test_dl = create_dataloader(
        DatasetConfig(
            name="object_centric",
            data_dir="data/mops_data/mops_object",
        ),
        batch_size=BATCH_SIZE,
        augment=False,
    )

    # Initialize the Trainer and run validation
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        fast_dev_run=False,
        logger=False,  # Disable logging for this test
        enable_checkpointing=False,
    )

    print("Starting CLIP classifier validation...")
    trainer.validate(model, dataloaders=test_dl)
    print("Validation complete.")

    # Example of running prediction
    print("\nStarting CLIP classifier prediction on a single batch...")
    for batch in test_dl:
        predictions = model.predict_step(batch, 0)
        # use matplotlib to show images with predicted labels
        import matplotlib.pyplot as plt
        import torchvision.transforms.functional as TF

        images = batch["image"]
        fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
        for i, (img, pred) in enumerate(zip(images, predictions)):
            img = TF.to_pil_image(img)
            axs[i].imshow(img)
            axs[i].set_title(CLASS_NAMES[pred])
            axs[i].axis("off")
        plt.show()

        print("Input labels:", batch["class_label"])
        print("Predicted labels:", predictions)
        break
    print("Prediction example complete.")


if __name__ == "__main__":
    test_clip_classifier()
