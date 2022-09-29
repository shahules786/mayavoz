import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import TQDMProgressBar


@hydra.main(config_path="train_config",config_name="config")
def main(config: DictConfig):

    callbacks = []
    callbacks.append(TQDMProgressBar(refresh_rate=10))
    logger = MLFlowLogger(experiment_name=config.mlflow.experiment_name,
                            run_name=config.mlflow.run_name, tags={"JOB_ID":os.environ.get("SLURM_JOBID")})


    parameters = config.hyperparameters

    dataset = instantiate(config.dataset)
    model = instantiate(config.model,dataset=dataset,lr=parameters.get("lr"),
            loss=parameters.get("loss"), metric = parameters.get("metric"))

    checkpoint = ModelCheckpoint(
        dirpath="",filename="model",monitor=parameters.get("loss"),verbose=False,
        mode="min",every_n_epochs=1
    )
    callbacks.append(checkpoint)
    early_stopping = EarlyStopping(
            monitor=parameters.get("loss"),
            mode="min",
            min_delta=0.0,
            patience=100,
            strict=True,
            verbose=False,
        )
    callbacks.append(early_stopping)

    trainer = instantiate(config.trainer,logger=logger,callbacks=callbacks)
    trainer.fit(model)



if __name__=="__main__":
    main()