from importlib import import_module
from huggingface_hub import cached_download, hf_hub_url
import numpy as np
import os
from typing import Optional, Union, List, Text, Dict, Any
from torch.optim import Adam
import torch
from torch.nn.functional import pad
import pytorch_lightning as pl
from pytorch_lightning.utilities.cloud_io import load as pl_load
from urllib.parse import urlparse
from pathlib import Path


from enhancer import __version__
from enhancer.data.dataset import Dataset
from enhancer.utils.io import Audio
from enhancer.utils.loss import Avergeloss
from enhancer.inference import Inference

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
        duration:Optional[float]=None,
        loss: Union[str, List] = "mse",
        metric:Union[str,List] = "mse"
    ):
        super().__init__()
        assert num_channels ==1 , "Enhancer only support for mono channel models"
        self.dataset = dataset
        self.save_hyperparameters("num_channels","sampling_rate","lr","loss","metric","duration")
        
    
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

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        pass


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


        return model 

    def infer_batch(self,batch,batch_size):
        
        assert batch.ndim == 3, f"Expected batch with 3 dimensions (batch,channels,samples) got only {batch.ndim}"
        batch_predictions = []
        self.eval().to(self.device)

        for batch_id in range(batch.shape[0],batch_size):
            batch_data = batch[batch_id:batch_id+batch_size,:,:].to(self.device)
            prediction = self(batch_data)
            batch_predictions.append(prediction)
        
        return torch.vstack(batch_predictions)

    def enhance(
        self,
        audio:Union[Path,np.ndarray,torch.Tensor],
        sampling_rate:Optional[int]=None,
        batch_size:int=32,
        save_output:bool=False,
        duration:Optional[int]=None,
        step_size:Optional[int]=None,):

        model_sampling_rate = self.model.hprams("sampling_rate")
        if duration is None:
            duration = self.model.hparams("duration")
        waveform = Inference.read_input(audio,sampling_rate,model_sampling_rate)
        waveform.to(self.device)
        window_size = round(duration * model_sampling_rate)
        batched_waveform = Inference.batchify(waveform,window_size,step_size=step_size)
        batch_prediction = self.infer_batch(batched_waveform,batch_size=batch_size)
        waveform = Inference.aggreagate(batch_prediction,window_size,step_size)
        
        if save_output and isinstance(audio,(str,Path)):
            Inference.write_output(waveform,audio,model_sampling_rate)

        else:
            return waveform            





        


       


            


        