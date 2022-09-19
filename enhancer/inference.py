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
from enhancer.utils.config import DEFAULT_DEVICE

class Inference:

    @staticmethod
    def read_input(audio, sr, model_sr):

        if isinstance(audio,(np.ndarray,torch.Tensor)):
            assert sr is not None, "Invalid sampling rate!"

        if isinstance(audio,str):
            audio = Path(audio)
            if not audio.is_file():
                raise ValueError(f"Input file {audio} does not exist")
            else:
                audio,sr = load_audio(audio,sr=sr,)
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
        Wrap into tensors of specified batch size
        """
        assert waveform.ndim == 2, f"Expcted input waveform with 2 dimensions (channels,samples), got {waveform.ndim}"
        _,num_samples = waveform.shape
        waveform = waveform.unsqueeze(0)
        step_size = window_size//2 if step_size is None else step_size

        if num_samples >= window_size:
            waveform_batch = F.unfold(waveform[None,...], kernel_size=(window_size,1),
            stride=(step_size,1), padding=(window_size,0))
            waveform_batch  = waveform_batch.permute(2,0,1)
        
        
        return waveform_batch


    def aggreagate(self,data:torch.Tensor,window_size:int, step_size:Optional[int]=None):
        """
        takes input as tensor outputs aggregated waveform
        """
        batch_size,n_channels,num_frames = data.shape
        window = get_window(window=window,Nx=data.shape[-1])
        window = torch.from_numpy(window).to(data.device)
        data *= window

        data = data.permute(1,2,0)
        data = F.fold(data,
            (num_frames,1),
            kernel_size=(window_size,1),
            stride=(step_size,1),
            padding=(window_size,0))

        return data

    @staticmethod
    def write_output(waveform:torch.Tensor,filename:Union[str,Path],sr:int):

        if isinstance(filename,str):
            filename = Path(filename)
        if filename.is_file():
            raise FileExistsError(f"file {filename} already exists")
        else:
            wavfile.write(filename,rate=sr,data=waveform.detach().cpu())
            








        


    