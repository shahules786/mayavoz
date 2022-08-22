
from genericpath import isdir
import librosa
import os
from torch.utils.data import IterableDataset
import torch


class Vctk(IterableDataset):
    """Dataset object for Voice Bank Corpus (VCTK) Dataset"""

    def __init__(self,clean_path,noisy_path,sample_length=1,num_samples=None):
        
        if not os.path.isdir(clean_path):
            raise ValueError(f"{clean_path} is not a valid directory")

        if not os.path.isdir(noisy_path):
            raise ValueError(f"{clean_path} is not a valid directory")

        self.clean_path = clean_path
        self.noisy_path = noisy_path

        if num_samples is None:
            self.num_samples = len([file for file in os.listdir(clean_path) if file.endswith(".wav")])
        else:
            self.num_samples = num_samples

        self.sample_length = max(0.1,sample_length)

    def __iter__(self):

        


        pass

    def __len__(self):
        pass
