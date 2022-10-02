import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
os.environ["HYDRA_FULL_ERROR"] = "1"
JOB_ID = os.environ.get("SLURM_JOBID")

@hydra.main(config_path="train_config",config_name="config")
def main(config: DictConfig):

    callbacks = []
    logger = MLFlowLogger(experiment_name=config.mlflow.experiment_name,
                            run_name=config.mlflow.run_name, tags={"JOB_ID":JOB_ID})


    parameters = config.hyperparameters

    dataset = instantiate(config.dataset)
    model = instantiate(config.model,dataset=dataset,lr=parameters.get("lr"),
            loss=parameters.get("loss"), metric = parameters.get("metric"))

    direction = model.valid_monitor
    checkpoint = ModelCheckpoint(
        dirpath="./model",filename=f"model_{JOB_ID}",monitor="val_loss",verbose=False,
        mode=direction,every_n_epochs=1
    )
    callbacks.append(checkpoint)
    early_stopping = EarlyStopping(
            monitor="val_loss",
            mode=direction,
            min_delta=0.0,
            patience=10,
            strict=True,
            verbose=False,
        )
    callbacks.append(early_stopping)

    trainer = instantiate(config.trainer,logger=logger,callbacks=callbacks)
    trainer.fit(model)
    logger.experiment.log_artifact(logger.run_id,f"./model/model_{JOB_ID}")



if __name__=="__main__":
    main()