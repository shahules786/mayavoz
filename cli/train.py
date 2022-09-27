import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from omegaconf import DictConfig,OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import torch


from enhancer.models.demucs import Demucs
from enhancer.data.dataset import EnhancerDataset


@hydra.main(config_path="train_config",config_name="config")
def main(config: DictConfig):

    logger = MLFlowLogger(experiment_name=config.mlflow.experiment_name,
                            run_name=config.mlflow.run_name)


    parameters = config.hyperparameters

    dataset = instantiate(config.dataset)
    model = instantiate(config.model,dataset=dataset,lr=parameters.get("lr"),
            loss=parameters.get("loss"), metric = parameters.get("metric"))

    trainer = instantiate(config.trainer,logger=logger)
    trainer.fit(model)



if __name__=="__main__":
    main()