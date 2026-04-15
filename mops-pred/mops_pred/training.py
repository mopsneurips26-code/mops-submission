import dataclasses

import draccus
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from mops_pred.config import ExperimentConfig
from mops_pred.datasets.dataset_factory import create_dataloader
from mops_pred.models import model_factory


def run_training(cfg: ExperimentConfig, model, debug=False):
    """Run the training loop with W&B logging and periodic checkpointing."""
    torch.set_float32_matmul_precision("medium")
    L.seed_everything(cfg.seed)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        offline=debug,
        config=dataclasses.asdict(cfg),
    )

    train_dl, test_dl = create_dataloader(cfg.dataset, batch_size=cfg.training.batch_size)

    print(wandb_logger.experiment.dir)

    periodic_checkpoint = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        filename="epoch-{epoch:03d}",
        every_n_epochs=20,
        save_top_k=-1,
        auto_insert_metric_name=False,
    )

    last_checkpoint = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        filename="last",
        save_last=True,
        save_top_k=0,
        auto_insert_metric_name=False,
    )

    trainer = L.Trainer(
        max_epochs=cfg.training.num_epochs,
        fast_dev_run=debug,
        callbacks=[periodic_checkpoint, last_checkpoint],
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=test_dl,
    )


@draccus.wrap()
def main(cfg: ExperimentConfig):
    debug = cfg.mode == "debug"

    torch.manual_seed(cfg.seed)
    model = model_factory.create_model(cfg.model)
    model.cuda()

    if cfg.checkpoint is not None:
        model.load_state_dict(torch.load(cfg.checkpoint)["model_state_dict"])

    run_training(cfg, model, debug=debug)
    wandb.finish()


if __name__ == "__main__":
    main()
