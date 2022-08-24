import os
import pytorch_lightning as pl
from typing import Optional

from enhancer.data.vctk import Vctk
from enhancer.utils.config import Files

DATASETS = ["Vctk"]

class Dataset(pl.LightningDataModule):

    def __init__(self,name:str, directory:str, files:Files, 
                    duration:float=1.0, sampling_rate:int=48000):
        super().__init__()

        self.train_clean = os.path.join(directory,Files.train_clean)
        self.train_noisy = os.path.join(directory,Files.train_noisy)
        self.valid_clean = os.path.join(directory,Files.test_clean)
        self.valid_noisy = os.path.join(directory,Files.test_noisy)

        if name.title() in DATASETS:
            self.data_obj = eval(name.title)

        self.duration = duration
        self.sampling_rate = sampling_rate

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self.data_obj()

    def train_loader(self):
        pass

    def valid_loader(self):
        pass
