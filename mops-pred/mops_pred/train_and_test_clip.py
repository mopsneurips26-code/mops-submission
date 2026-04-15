import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from mops_pred.config import DatasetConfig
from mops_pred.datasets.dataset_factory import create_dataloader
from mops_pred.models.object_classifier import CLIPObjectClassifier

# Hardcoded constants for training and testing
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
BATCH_SIZE = 32
NUM_EPOCHS = 10
IMAGE_SIZE = 224
NUM_CLASSES = len(CLASS_NAMES)


def train_and_test_clip_classifier():
    """
    Initializes, fine-tunes, and tests the CLIPObjectClassifier.
    """
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(42)

    # Initialize the model
    model = CLIPObjectClassifier(model_name=MODEL_NAME, class_names=CLASS_NAMES)

    # Create dataloaders
    train_dl, test_dl = create_dataloader(
        DatasetConfig(
            name="object_centric",
            data_dir="data/mops_data/mops_object",
        ),
        batch_size=BATCH_SIZE,
        augment=True,
    )

    # Checkpoint to save the best model based on validation accuracy
    checkpoint_callback = ModelCheckpoint(
        monitor="val/acc",
        dirpath="checkpoints",
        filename="clip-best-model",
        save_top_k=1,
        mode="max",
    )

    # Initialize the Trainer
    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        logger=True,  # Enable default logging
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    print("Starting CLIP classifier fine-tuning...")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=test_dl)
    print("Fine-tuning complete.")

    print("\nLoading best model and running final validation...")
    best_model = CLIPObjectClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    trainer.validate(best_model, dataloaders=test_dl)
    print("Final validation complete.")

    # Example of running prediction with the fine-tuned model
    print("\nStarting CLIP classifier prediction on a single batch...")
    for batch in test_dl:
        # Move batch to the correct device
        batch = {k: v.to(best_model.device) for k, v in batch.items()}
        predictions = best_model.predict_step(batch, 0)

        # use matplotlib to show images with predicted labels
        import matplotlib.pyplot as plt
        import torchvision.transforms.functional as TF

        images = batch["image"].cpu()
        fig, axs = plt.subplots(1, len(images), figsize=(20, 5))
        for i, (img, pred) in enumerate(zip(images, predictions)):
            # The dataloader returns normalized images, so we need to unnormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            img = TF.to_pil_image(img)

            axs[i].imshow(img)
            axs[i].set_title(f"Pred: {CLASS_NAMES[pred]}")
            axs[i].axis("off")
        plt.show()

        print("Input labels:", batch["class_label"].cpu().numpy())
        print("Predicted labels:", predictions.cpu().numpy())
        break
    print("Prediction example complete.")


if __name__ == "__main__":
    train_and_test_clip_classifier()
