import os
from types import MethodType

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

    dataset = instantiate(config.dataset)
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
        monitor="val_loss",
        verbose=False,
        mode=direction,
        every_n_epochs=1,
    )
    callbacks.append(checkpoint)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode=direction,
        min_delta=0.0,
        patience=parameters.get("EarlyStopping_patience", 10),
        strict=True,
        verbose=False,
    )
    callbacks.append(early_stopping)

    def configure_optimizer(self):
        optimizer = instantiate(
            config.optimizer,
            lr=parameters.get("lr"),
            parameters=self.parameters(),
        )
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode=direction,
            factor=parameters.get("ReduceLr_factor", 0.1),
            verbose=True,
            min_lr=parameters.get("min_lr", 1e-6),
            patience=parameters.get("ReduceLr_patience", 3),
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    model.configure_parameters = MethodType(configure_optimizer, model)

    trainer = instantiate(config.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model)
    trainer.test(model)

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
