from importlib import import_module
from huggingface_hub import cached_download, hf_hub_url
import os
from typing import Optional, Union, List, Path, Text
from torch.optim import Adam
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load
from urllib.parse import urlparse


from enhancer import __version__
from enhancer.data.dataset import Dataset
from enhancer.utils.loss import Avergeloss

CACHE_DIR = ""
HF_TORCH_WEIGHTS = ""
DEFAULT_DEVICE = "cpu"

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
    def from_pretrained(
        cls,
        checkpoint: Union[Path, Text],
        map_location = None,
        hparams_file: Union[Path, Text] = None,
        strict: bool = True,
        use_auth_token: Union[Text, None] = None,
        cached_dir: Union[Path, Text]=CACHE_DIR,
        **kwargs
    ):

        checkpoint = str(checkpoint)
        if hparams_file is not None:
            hparams_file = str(hparams_file)

        if os.path.isfile(checkpoint):
            model_path_pl = checkpoint
        elif urlparse(checkpoint).scheme in ("http","https"):
            model_path_pl = checkpoint
        else:
            
            if "@" in checkpoint:
                model_id = checkpoint.split("@")[0]
                revision_id = checkpoint.split("@")[1]
            else:
                model_id = checkpoint
                revision_id = None
            
            url = hf_hub_url(
                model_id,filename=HF_TORCH_WEIGHTS,revision=revision_id
            )
            model_path_pl = cached_download(
                url=url,library_name="enhancer",library_version=__version__,
                cache_dir=cached_dir,use_auth_token=use_auth_token
            )

        if map_location is None:
            map_location = torch.device(DEFAULT_DEVICE)

        loaded_checkpoint = pl_load(model_path_pl,map_location)
        module_name = loaded_checkpoint["architecture"]["module"]
        class_name =  loaded_checkpoint["architecture"]["class"]
        module = import_module(module_name)
        Klass = getattr(module, class_name)

        try:
            model = Klass.load_from_checkpoint(
                checkpoint_path = model_path_pl,
                map_location = map_location,
                hparams_file = hparams_file,
                strict = strict,
                **kwargs
            )
        except Exception as e:
            print(e)


        

            


        