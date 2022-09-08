from typing import Optional
from torch.optim import Adam
import pytorch_lightning as pl

from enhancer.data.dataset import Dataset


class Model(pl.LightningModule):

    def __init__(
        self,
        num_channels:int=1,
        sampling_rate:int=16000,
        lr:float=1e-3,
        dataset:Optional[Dataset]=None,
    ):
        super().__init__()
        assert num_channels ==1 , "Enhancer only support for mono channel models"
        self.save_hyperparameters("num_channels","sampling_rate","lr")
        self.dataset = dataset

    
    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self,dataset):
        self._dataset = dataset

    def setup(self,stage:Optional[str]=None):
        if stage == "fit":
            self.dataset.setup(stage)
            self.dataset.model = self 
        

    def train_dataloader(self):
        return self.dataset.train_dataloader()

    def val_dataloader(self):
        return self.dataset.val_dataloader()

    def configure_optimizers(self):
        return Adam(self.parameters, lr = self.hparams.lr)

    def training_step(self,batch, batch_idx:int):
        pass

    @classmethod
    def from_pretrained(cls,):
        pass