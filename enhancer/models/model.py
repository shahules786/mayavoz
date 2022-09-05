from typing import Optional
import pytorch_lightning as pl

from enhancer.data.dataset import Dataset


class Model(pl.LightningModule):

    def __init__(
        self,
        dataset:Dataset
    ):
        super().__init__()
        self.dataset = dataset

        pass
    
    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self,dataset):
        self._dataset = dataset

    def setup(
        self,
        stage:Optional[str]=None
    ):
        if stage == "fit":
            self.dataset.setup(stage)
            self.dataset.model = self 
        

    def train_dataloader(
        self
    ):
        return self.dataset.train_dataloader()

    def val_dataloader(
        self
    ):
        return self.dataset.val_dataloader()
