
import glob
import math
import numpy as np
import os
from scipy.io import wavfile
from torch.utils.data import IterableDataset
import torch.nn.functional as F

from enhancer.utils.random import create_unique_rng
from enhancer.utils.io import Audio



class VctkDataset:

    def __init__(self):
        pass

    def train_loader(self):
        pass

    def valid_loader(self):
        pass

    def test_loader(self):
        pass



class Vctk(IterableDataset):
    """Dataset object for Voice Bank Corpus (VCTK) Dataset"""

    def __init__(self,clean_path,noisy_path,duration=1.0,sampling_rate=48000):
        
        if not os.path.isdir(clean_path):
            raise ValueError(f"{clean_path} is not a valid directory")

        if not os.path.isdir(noisy_path):
            raise ValueError(f"{clean_path} is not a valid directory")

        self.sampling_rate = sampling_rate
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.files_duration = self.get_matching_files_duration()
        self.wav_samples = list(self.files_duration.keys())
        self.duration = max(1.0,duration)
        self.audio = Audio(self.sampling_rate,mono=True,return_tensor=True)

    def get_matching_files_duration(self):

        matching_wavfiles_dur = dict()
        clean_filenames = [file.split('/')[-1] for file in glob.glob(os.path.join(self.clean_path,"*.wav"))]
        noisy_filenames = [file.split('/')[-1] for file in glob.glob(os.path.join(self.noisy_path,"*.wav"))]
        common_filenames = np.intersect1d(noisy_filenames,clean_filenames)

        for file_name in common_filenames:

             sr_clean, clean_file = wavfile.read(os.path.join(self.clean_path,file_name))
             sr_noisy, noisy_file = wavfile.read(os.path.join(self.noisy_path,file_name))
             if ((clean_file.shape[-1]==noisy_file.shape[-1]) and 
                    (sr_clean==self.sampling_rate) and 
                        (sr_noisy==self.sampling_rate)):
                matching_wavfiles_dur.update({file_name:(clean_file.shape[-1]/self.sampling_rate)})

        return matching_wavfiles_dur

    def __iter__(self):

        rng = create_unique_rng(12) ##pass epoch number here
        
        while True:

            file_name,*_ = rng.choices(self.wav_samples,k=1,
                        weights=[self.files_duration[file] for file in self.wav_samples])
            file_duration = self.files_duration.get(file_name)
            start_time = round(rng.uniform(0,file_duration- self.duration),2)
            data = self.prepare_segment(file_name,start_time)
            yield data

    def prepare_segment(self,file_name:str, start_time:float):

        clean_segment = self.audio(os.path.join(self.clean_path,file_name),
                                    offset=start_time,duration=self.duration)
        noisy_segment = self.audio(os.path.join(self.noisy_path,file_name),
                                    offset=start_time,duration=self.duration)
        clean_segment = F.pad(clean_segment,(0,int(self.duration*self.sampling_rate-clean_segment.shape[-1])))
        noisy_segment = F.pad(noisy_segment,(0,int(self.duration*self.sampling_rate-noisy_segment.shape[-1])))
        return {"clean": clean_segment,"noisy":noisy_segment}
        
    def __len__(self):

        return math.ceil(sum(self.files_duration.values())/self.duration)


        
