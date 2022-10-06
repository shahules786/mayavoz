import os
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import torch
import torchaudio


class Audio:
    """
    Audio utils
    parameters:
        sampling_rate : int, defaults to 16KHz
            audio sampling rate
        mono: bool, defaults to True
        return_tensors: bool, defaults to True
            returns torch tensor type if set to True else numpy ndarray
    """

    def __init__(
        self, sampling_rate: int = 16000, mono: bool = True, return_tensor=True
    ) -> None:

        self.sampling_rate = sampling_rate
        self.mono = mono
        self.return_tensor = return_tensor

    def __call__(
        self,
        audio: Union[Path, np.ndarray, torch.Tensor],
        sampling_rate: Optional[int] = None,
        offset: Optional[float] = None,
        duration: Optional[float] = None,
    ):
        """
        read and process input audio
        parameters:
            audio: Path to audio file or numpy array or torch tensor
                single input audio
            sampling_rate : int, optional
                sampling rate of the audio input
            offset: float, optional
                offset from which the audio must be read, reads from beginning if unused.
            duration: float (seconds), optional
                read duration, reads full audio starting from offset if not used
        """
        if isinstance(audio, str):
            if os.path.exists(audio):
                audio, sampling_rate = librosa.load(
                    audio,
                    sr=sampling_rate,
                    mono=False,
                    offset=offset,
                    duration=duration,
                )
                if len(audio.shape) == 1:
                    audio = audio.reshape(1, -1)
            else:
                raise FileNotFoundError(f"File {audio} deos not exist")
        elif isinstance(audio, np.ndarray):
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)
        else:
            raise ValueError("audio should be either filepath or numpy ndarray")

        if self.mono:
            audio = self.convert_mono(audio)

        if sampling_rate:
            audio = self.__class__.resample_audio(
                audio, self.sampling_rate, sampling_rate
            )
        if self.return_tensor:
            return torch.tensor(audio)
        else:
            return audio

    @staticmethod
    def convert_mono(audio: Union[np.ndarray, torch.Tensor]):
        """
        convert input audio into mono (1)
        parameters:
            audio: np.ndarray or torch.Tensor
        """
        if len(audio.shape) > 2:
            assert (
                audio.shape[0] == 1
            ), "convert mono only accepts single waveform"
            audio = audio.reshape(audio.shape[1], audio.shape[2])

        assert (
            audio.shape[1] >> audio.shape[0]
        ), f"expected input format (num_channels,num_samples) got {audio.shape}"
        num_channels, num_samples = audio.shape
        if num_channels > 1:
            return audio.mean(axis=0).reshape(1, num_samples)
        return audio

    @staticmethod
    def resample_audio(
        audio: Union[np.ndarray, torch.Tensor], sr: int, target_sr: int
    ):
        """
        resample audio to desired sampling rate
        parameters:
            audio : Path to audio file or numpy array or torch tensor
                audio waveform
            sr : int
                current sampling rate
            target_sr : int
                target sampling rate

        """
        if sr != target_sr:
            if isinstance(audio, np.ndarray):
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            elif isinstance(audio, torch.Tensor):
                audio = torchaudio.functional.resample(
                    audio, orig_freq=sr, new_freq=target_sr
                )
            else:
                raise ValueError(
                    "Input should be either numpy array or torch tensor"
                )

        return audio
