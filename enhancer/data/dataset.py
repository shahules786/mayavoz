import multiprocessing
import math
import os
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader, Dataset
import torch.nn.functional as F
from typing import Optional

from enhancer.data.fileprocessor import Fileprocessor
from enhancer.utils.random import create_unique_rng
from enhancer.utils.io import Audio
from enhancer.utils import check_files
from enhancer.utils.config import Files

class TrainDataset(IterableDataset):
    
    def __init__(self,dataset):
        self.dataset = dataset

    def __iter__(self):
        return self.dataset.train__iter__()

    def __len__(self):
        return self.dataset.train__len__()

class ValidDataset(Dataset):
    
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self,idx):
        return self.dataset.val__getitem__(idx)

    def __len__(self):
        return self.dataset.val__len__()

class TaskDataset(pl.LightningDataModule):

    def __init__(
        self,
        name:str,
        root_dir:str, 
        files:Files, 
        duration:float=1.0,
        sampling_rate:int=48000,
        matching_function = None,
        batch_size=32,
        num_workers:Optional[int]=None):
        super().__init__()

        self.name = name
        self.files,self.root_dir = check_files(root_dir,files)
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.matching_function = matching_function
        self._validation = []
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()//2
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):

        if stage in ("fit",None):

            train_clean = os.path.join(self.root_dir,self.files.train_clean)
            train_noisy = os.path.join(self.root_dir,self.files.train_noisy)
            fp = Fileprocessor.from_name(self.name,train_clean,
                                        train_noisy, self.matching_function)
            self.train_data = fp.prepare_matching_dict()
            
            val_clean = os.path.join(self.root_dir,self.files.test_clean)
            val_noisy = os.path.join(self.root_dir,self.files.test_noisy)
            fp =  Fileprocessor.from_name(self.name,val_clean,
                                        val_noisy, self.matching_function)
            val_data = fp.prepare_matching_dict()

            for item in val_data:
                clean,noisy,total_dur = item.values()
                if total_dur < self.duration:
                    continue
                num_segments = round(total_dur/self.duration)
                for index in range(num_segments):
                    start_time = index * self.duration
                    self._validation.append(({"clean":clean,"noisy":noisy},
                                            start_time))
    def train_dataloader(self):
        return DataLoader(TrainDataset(self), batch_size = self.batch_size,num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(ValidDataset(self), batch_size = self.batch_size,num_workers=self.num_workers)

class EnhancerDataset(TaskDataset):
    """
    Dataset object for creating clean-noisy speech enhancement datasets
    paramters:
    name : str
        name of the dataset 
    root_dir : str
        root directory of the dataset containing clean/noisy folders
    files : Files
        dataclass containing train_clean, train_noisy, test_clean, test_noisy 
        folder names (refer cli/train_config/dataset)
    duration : float
        expected audio duration of single audio sample for training
    sampling_rate : int
        desired sampling rate 
    batch_size : int
        batch size of each batch
    num_workers : int
        num workers to be used while training
    matching_function : str
        maching functions - (one_to_one,one_to_many). Default set to None.
        use one_to_one mapping for datasets with one noisy file for each clean file
        use one_to_many mapping for multiple noisy files for each clean file

    
    """

    def __init__(
        self,
        name:str,
        root_dir:str,
        files:Files,
        duration=1.0,
        sampling_rate=48000,
        matching_function=None,
        batch_size=32,
        num_workers:Optional[int]=None):
        
        super().__init__(
            name=name,
            root_dir=root_dir,
            files=files,
            sampling_rate=sampling_rate,
            duration=duration,
            matching_function = matching_function,
            batch_size=batch_size,
            num_workers = num_workers,

        )

        self.sampling_rate = sampling_rate
        self.files = files
        self.duration = max(1.0,duration)
        self.audio = Audio(self.sampling_rate,mono=True,return_tensor=True)

    def setup(self, stage:Optional[str]=None):
        
        super().setup(stage=stage)

    def train__iter__(self):

        rng = create_unique_rng(self.model.current_epoch) 
        
        while True:

            file_dict,*_ = rng.choices(self.train_data,k=1,
                        weights=[file["duration"] for file in self.train_data])
            file_duration = file_dict['duration']
            start_time = round(rng.uniform(0,file_duration- self.duration),2)
            data = self.prepare_segment(file_dict,start_time)
            yield data

    def val__getitem__(self,idx):
        return self.prepare_segment(*self._validation[idx])
        
    def prepare_segment(self,file_dict:dict, start_time:float):

        clean_segment = self.audio(file_dict["clean"],
                                    offset=start_time,duration=self.duration)
        noisy_segment = self.audio(file_dict["noisy"],
                                    offset=start_time,duration=self.duration)
        clean_segment = F.pad(clean_segment,(0,int(self.duration*self.sampling_rate-clean_segment.shape[-1])))
        noisy_segment = F.pad(noisy_segment,(0,int(self.duration*self.sampling_rate-noisy_segment.shape[-1])))
        return {"clean": clean_segment,"noisy":noisy_segment}
        
    def train__len__(self):
        return math.ceil(sum([file["duration"] for file in self.train_data])/self.duration)

    def val__len__(self):
        return len(self._validation)


