import os
from types import MethodType

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from torch_audiomentations import Compose, Shift

os.environ["HYDRA_FULL_ERROR"] = "1"
JOB_ID = os.environ.get("SLURM_JOBID", "0")


@hydra.main(config_path="train_config", config_name="config")
def main(config: DictConfig):

    OmegaConf.save(config, "config_log.yaml")

    callbacks = []
    logger = MLFlowLogger(
        experiment_name=config.mlflow.experiment_name,
        run_name=config.mlflow.run_name,
        tags={"JOB_ID": JOB_ID},
    )

    parameters = config.hyperparameters
    # apply_augmentations = Compose(
    #     [
    #         Shift(min_shift=0.5, max_shift=1.0, shift_unit="seconds", p=0.5),
    #     ]
    # )

    dataset = instantiate(config.dataset, augmentations=None)
    model = instantiate(
        config.model,
        dataset=dataset,
        lr=parameters.get("lr"),
        loss=parameters.get("loss"),
        metric=parameters.get("metric"),
    )

    direction = model.valid_monitor
    checkpoint = ModelCheckpoint(
        dirpath="./model",
        filename=f"model_{JOB_ID}",
        monitor="valid_loss",
        verbose=False,
        mode=direction,
        every_n_epochs=1,
    )
    callbacks.append(checkpoint)
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    if parameters.get("Early_stop", False):
        early_stopping = EarlyStopping(
            monitor=f"valid_{parameters.get('EarlyStopping_metric','loss')}",
            mode=direction,
            min_delta=parameters.get("EarlyStopping_delta", 0.00),
            patience=parameters.get("EarlyStopping_patience", 10),
            strict=True,
            verbose=False,
        )
        callbacks.append(early_stopping)

    def configure_optimizers(self):
        optimizer = instantiate(
            config.optimizer,
            lr=parameters.get("lr"),
            params=self.parameters(),
        )
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode=direction,
            factor=parameters.get("ReduceLr_factor", 0.1),
            verbose=True,
            min_lr=parameters.get("min_lr", 1e-6),
            patience=parameters.get("ReduceLr_patience", 3),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": f'valid_{parameters.get("ReduceLr_monitor", "loss")}',
        }

    model.configure_optimizers = MethodType(configure_optimizers, model)

    trainer = instantiate(config.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model)
    trainer.test(ckpt_path="best")

    logger.experiment.log_artifact(
        logger.run_id, f"{trainer.default_root_dir}/config_log.yaml"
    )

    saved_location = os.path.join(
        trainer.default_root_dir, "model", f"model_{JOB_ID}.ckpt"
    )
    if os.path.isfile(saved_location):
        logger.experiment.log_artifact(logger.run_id, saved_location)
        logger.experiment.log_param(
            logger.run_id,
            "num_train_steps_per_epoch",
            dataset.train__len__() / dataset.batch_size,
        )
        logger.experiment.log_param(
            logger.run_id,
            "num_valid_steps_per_epoch",
            dataset.val__len__() / dataset.batch_size,
        )


if __name__ == "__main__":
    main()
