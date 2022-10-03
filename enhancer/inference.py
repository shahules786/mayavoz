from json import load
import wave
import numpy as np
from scipy.signal import get_window
from scipy.io import wavfile
from typing import List, Optional, Union
import torch
import torch.nn.functional as F
from pathlib import Path
from librosa import load as load_audio

from enhancer.utils import Audio

class Inference:

    @staticmethod
    def read_input(audio, sr, model_sr):

        if isinstance(audio,(np.ndarray,torch.Tensor)):
            assert sr is not None, "Invalid sampling rate!"
            if len(audio.shape) == 1:
                audio = audio.reshape(1,-1)

        if isinstance(audio,str):
            audio = Path(audio)
            if not audio.is_file():
                raise ValueError(f"Input file {audio} does not exist")
            else:
                audio,sr = load_audio(audio,sr=sr,)
                if len(audio.shape) == 1:
                    audio = audio.reshape(1,-1)
        else:
            assert audio.shape[0] == 1, "Enhance inference only supports single waveform"

        waveform = Audio.resample_audio(audio,sr=sr,target_sr=model_sr)
        waveform = Audio.convert_mono(waveform)
        if isinstance(waveform,np.ndarray):
            waveform = torch.from_numpy(waveform)

        return waveform

    @staticmethod
    def batchify(waveform: torch.Tensor, window_size:int, step_size:Optional[int]=None):
        """
        break input waveform into samples with duration specified. 
        """
        assert waveform.ndim == 2, f"Expcted input waveform with 2 dimensions (channels,samples), got {waveform.ndim}"
        _,num_samples = waveform.shape
        waveform = waveform.unsqueeze(-1)
        step_size = window_size//2 if step_size is None else step_size

        if num_samples >= window_size:
            waveform_batch = F.unfold(waveform[None,...], kernel_size=(window_size,1),
            stride=(step_size,1), padding=(window_size,0))
            waveform_batch  = waveform_batch.permute(2,0,1)
        
        
        return waveform_batch

    @staticmethod
    def aggreagate(data:torch.Tensor,window_size:int,total_frames:int,step_size:Optional[int]=None,
         window="hanning",):
        """
        takes input as tensor outputs aggregated waveform
        """
        num_chunks,n_channels,num_frames = data.shape
        window = get_window(window=window,Nx=data.shape[-1])
        window = torch.from_numpy(window).to(data.device)
        data *= window
        step_size = window_size//2 if step_size is None else step_size


        data = data.permute(1,2,0)
        data = F.fold(data,
            (total_frames,1),
            kernel_size=(window_size,1),
            stride=(step_size,1),
            padding=(window_size,0)).squeeze(-1)

        return data.reshape(1,n_channels,-1)

    @staticmethod
    def write_output(waveform:torch.Tensor,filename:Union[str,Path],sr:int):

        if isinstance(filename,str):
            filename = Path(filename)

        parent, name = filename.parent, "cleaned_"+filename.name
        filename = parent/Path(name)
        if filename.is_file():
            raise FileExistsError(f"file {filename} already exists")
        else:
            if isinstance(waveform,torch.Tensor):
                waveform = waveform.detach().cpu().squeeze().numpy()
            wavfile.write(filename,rate=sr,data=waveform)

    @staticmethod
    def prepare_output(waveform:torch.Tensor, model_sampling_rate:int,
        audio:Union[str,np.ndarray,torch.Tensor], sampling_rate:Optional[int]
    ):
        if isinstance(audio,np.ndarray):
            waveform = waveform.detach().cpu().numpy()

        if sampling_rate!=None:
            waveform = Audio.resample_audio(waveform, sr=model_sampling_rate, target_sr=sampling_rate)

        return waveform     








        


    