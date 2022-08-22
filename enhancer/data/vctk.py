
import glob
import math
import os
from scipy.io import wavfile
from torch.utils.data import IterableDataset

from enhancer.utils.random import create_unique_rng
from enhancer.utils.io import Audio


class Vctk(IterableDataset):
    """Dataset object for Voice Bank Corpus (VCTK) Dataset"""

    def __init__(self,clean_path,noisy_path,duration=1,sampling_rate=16000,num_samples=None):
        
        if not os.path.isdir(clean_path):
            raise ValueError(f"{clean_path} is not a valid directory")

        if not os.path.isdir(noisy_path):
            raise ValueError(f"{clean_path} is not a valid directory")

        self.sampling_rate = sampling_rate
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.wav_samples =[file.split('/')[-1] for file in glob.glob(os.path.join(clean_path,"*.wav"))]

        if num_samples is None:
            self.num_samples = len(self.wav_samples)
        else:
            self.num_samples = num_samples

        self.duration = max(1.0,duration)
        self.audio = Audio(self.sampling_rate,mono=True,return_tensor=True)
        self.files_duration = self.get_files_duration()

    def get_file_duration(self):

        files_duration = {}
        for file in self.clean_path:
            wavfile = wavfile.read(os.path.join(self.clean_path,file),rate=self.sampling_rate)
            files_duration.update({file:math.ceil(wavfile/self.sampling_rate)})

        return files_duration


    def __iter__(self):

        rng = create_unique_rng(12) ##pass epoch number here

        while True:

            file_name = rng.choices(self.wav_samples,k=1)
            file_duration = self.files_duration.get(file_name)
            start_time = rng.randint(0,math.ceil(file_duration- self.duration))
            data = self.prepare_segment(file_name,start_time)
            yield data

    def prepare_segment(self,file_name:str, start_time:int):

        clean_segment = self.audio(os.path.join(self.clean_path,file_name),
                                    offset=start_time,duration=self.duration)
        noisy_segment = self.audio(os.path.join(self.noisy_path,file_name),
                                    offset=start_time,duration=self.duration)

        return {"clean": clean_segment,"noisy":noisy_segment}
        
    def __len__(self):

        return math.ceil(sum(self.files_duration.values())/self.duration)


        
