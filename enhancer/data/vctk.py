
import glob
import math
import numpy as np
import os
from scipy.io import wavfile
from torch.utils.data import IterableDataset
import torch.nn.functional as F

from enhancer.utils.random import create_unique_rng
from enhancer.utils.io import Audio
from enhancer.utils import Fileprocessor



class EnhancerDataset(IterableDataset):
    """Dataset object for creating clean-noisy speech enhancement datasets"""

    def __init__(self,name:str,clean_dir,noisy_dir,duration=1.0,sampling_rate=48000, matching_function=None):
        
        if not os.path.isdir(clean_dir):
            raise ValueError(f"{clean_dir} is not a valid directory")

        if not os.path.isdir(noisy_dir):
            raise ValueError(f"{clean_dir} is not a valid directory")

        self.sampling_rate = sampling_rate
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.duration = max(1.0,duration)
        self.audio = Audio(self.sampling_rate,mono=True,return_tensor=True)

        fp = Fileprocessor.from_name(name,clean_dir,noisy_dir,matching_function)
        self.valid_files = fp.prepare_matching_dict()

    def __iter__(self):

        rng = create_unique_rng(12) ##pass epoch number here
        
        while True:

            file_dict,*_ = rng.choices(self.valid_files,k=1,
                        weights=[self.valid_files[file]['duration'] for file in self.valid_files])
            file_duration = file_dict['duration']
            start_time = round(rng.uniform(0,file_duration- self.duration),2)
            data = self.prepare_segment(file_dict,start_time)
            yield data

    def prepare_segment(self,file_dict:dict, start_time:float):

        clean_segment = self.audio(file_dict.keys()[0],
                                    offset=start_time,duration=self.duration)
        noisy_segment = self.audio(file_dict['noisy'],
                                    offset=start_time,duration=self.duration)
        clean_segment = F.pad(clean_segment,(0,int(self.duration*self.sampling_rate-clean_segment.shape[-1])))
        noisy_segment = F.pad(noisy_segment,(0,int(self.duration*self.sampling_rate-noisy_segment.shape[-1])))
        return {"clean": clean_segment,"noisy":noisy_segment}
        
    def __len__(self):

        return math.ceil(sum([file["duration"] for file in self.valid_files])/self.duration)


        
