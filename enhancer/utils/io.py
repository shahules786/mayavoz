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

        if sampling_rate:
            audio =  self.__class__.resample_audio(audio,self.sampling_rate,sampling_rate)
        if self.return_tensor:
            return torch.tensor(audio)
        else:
            return audio

    @staticmethod
    def convert_mono(
        audio

    ):
        if len(audio.shape)>2:
            assert audio.shape[0] == 1, "convert mono only accepts single waveform"
            audio = audio.reshape(audio.shape[1],audio.shape[2])
         
        assert audio.shape[0] > audio.shape[1], "expected input format (num_channels,num_samples)"
        num_channels,num_samples = audio.shape
        if num_channels>1:
            return audio.mean(axis=0).reshape(1,num_samples)
        return audio


    @staticmethod
    def resample_audio(
        audio,
        sr:int,
        target_sr:int
    ):
        if sr!=target_sr:
            if isinstance(audio,np.ndarray):
                audio = librosa.resample(audio,orig_sr=sr,target_sr=target_sr)
            elif isinstance(audio,torch.Tensor):
                audio = torchaudio.functional.resample(audio,orig_freq=sr,new_freq=target_sr)
            else:
                raise ValueError("Input should be either numpy array or torch tensor")

        return audio
