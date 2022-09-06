import os
import librosa
from typing import Optional
from matplotlib.pyplot import axis
import numpy as np
import torch
import torchaudio

class Audio:

    def __init__(
        self,
        sampling_rate:int=16000,
        mono:bool=True,
        return_tensor=True
    ) -> None:
        
        self.sampling_rate = sampling_rate
        self.mono = mono
        self.return_tensor = return_tensor

    def __call__(
        self,
        audio,
        sampling_rate:Optional[int]=None,
        offset:Optional[float] = None,
        duration:Optional[float] = None
    ):
        if isinstance(audio,str):
            if os.path.exists(audio):
                audio,sampling_rate = librosa.load(audio,sr=sampling_rate,mono=False,
                offset=offset,duration=duration)
                if len(audio.shape) == 1:
                    audio = audio.reshape(1,-1)
            else:
                raise FileNotFoundError(f"File {audio} deos not exist")
        elif isinstance(audio,np.ndarray):
            if len(audio.shape) == 1:
                audio = audio.reshape(1,-1)
        else:
            raise ValueError("audio should be either filepath or numpy ndarray")

        if self.mono:
            audio = self.convert_mono(audio)

        resampled_audio =  self.__class__.resample_audio(audio,self.sampling_rate,sampling_rate)
        if self.return_tensor:
            return torch.tensor(resampled_audio)
        else:
            return resampled_audio

    def convert_mono(
        self,
        audio

    ):
        num_channels,num_samples = audio.shape
        if num_channels>1 and self.mono:
            return audio.mean(axis=0).reshape(1,num_samples)
        return audio


    @staticmethod
    def resample_audio(
        audio,
        sr:int,
        target_sr:int
    ):
        if sr!=target_sr:
            audio = librosa.resample(audio,orig_sr=sr,target_sr=target_sr)

        return audio

    @staticmethod
    def pt_resample_audio(
        audio,
        sr:int,
        target_sr:int
    ):
        if sr!=target_sr:
            audio = torchaudio.functional.resample(audio,orig_freq=sr,new_freq=target_sr)

        return audio