from typing import Optional, Union, List
from torch.optim import Adam
import pytorch_lightning as pl
import torch

from enhancer import __version__
from enhancer.data.dataset import Dataset
from enhancer.utils.loss import Avergeloss


class Model(pl.LightningModule):

    def __init__(
        self,
        num_channels:int=1,
        sampling_rate:int=16000,
        lr:float=1e-3,
        dataset:Optional[Dataset]=None,
        loss: Union[str, List] = "mse",
        metric:Union[str,List] = "mse"
    ):
        super().__init__()
        assert num_channels ==1 , "Enhancer only support for mono channel models"
        self.save_hyperparameters("num_channels","sampling_rate","lr","loss","metric")
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
            self.loss = self.setup_loss(self.hparams.loss) 
            self.metric = self.setup_loss(self.hparams.metric)
        
    def setup_loss(self,loss):

        if isinstance(loss,str):
            losses = [loss]
        
        return Avergeloss(losses)

    def train_dataloader(self):
        return self.dataset.train_dataloader()

    def val_dataloader(self):
        return self.dataset.val_dataloader()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr = self.hparams.lr)

    def training_step(self,batch, batch_idx:int):

        mixed_waveform = batch["noisy"]
        target = batch["clean"]
        prediction = self(mixed_waveform)

        loss = self.loss(prediction, target)

        return {"loss":loss}

    def validation_step(self,batch,batch_idx:int):

        mixed_waveform = batch["noisy"]
        target = batch["clean"]
        prediction = self(mixed_waveform)

        loss = self.metric(prediction, target)

        return {"loss":loss}

    def on_save_checkpoint(self, checkpoint):

        checkpoint["enhancer"] = {
            "version": {
                "enhancer":__version__,
                "pytorch":torch.__version__
            },
            "architecture":{
                "module":self.__class__.__module__,
                "class":self.__class__.__name__
            }

        }

    @classmethod
    def from_pretrained(cls,):
        pass