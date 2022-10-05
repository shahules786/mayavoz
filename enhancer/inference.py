import numpy as np
from scipy.signal import get_window
from scipy.io import wavfile
from typing import Optional, Union
import torch
import torch.nn.functional as F
from pathlib import Path
from librosa import load as load_audio

from enhancer.utils import Audio


class Inference:
    """
    contains methods used for inference.
    """

    @staticmethod
    def read_input(audio, sr, model_sr):
        """
        read and verify audio input regardless of the input format.
        arguments:
            audio : audio input
            sr : sampling rate of input audio
            model_sr : sampling rate used for model training.
        """

        if isinstance(audio, (np.ndarray, torch.Tensor)):
            assert sr is not None, "Invalid sampling rate!"

        if isinstance(audio, str):
            audio = Path(audio)
            if not audio.is_file():
                raise ValueError(f"Input file {audio} does not exist")
            else:
                audio, sr = load_audio(
                    audio,
                    sr=sr,
                )
                if len(audio.shape) == 1:
                    audio = audio.reshape(1, -1)
        else:
            assert (
                audio.shape[0] == 1
            ), "Enhance inference only supports single waveform"

        waveform = Audio.resample_audio(audio, sr=sr, target_sr=model_sr)
        waveform = Audio.convert_mono(waveform)
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        return waveform

    @staticmethod
    def batchify(
        waveform: torch.Tensor,
        window_size: int,
        step_size: Optional[int] = None,
    ):
        """
        break input waveform into samples with duration specified.(Overlap-add)
        arguments:
            waveform : audio waveform
            window_size : window size used for splitting waveform into batches
            step_size : step_size used for splitting waveform into batches
        """
        assert (
            waveform.ndim == 2
        ), f"Expcted input waveform with 2 dimensions (channels,samples), got {waveform.ndim}"
        _, num_samples = waveform.shape
        waveform = waveform.unsqueeze(-1)
        step_size = window_size // 2 if step_size is None else step_size

        if num_samples >= window_size:
            waveform_batch = F.unfold(
                waveform[None, ...],
                kernel_size=(window_size, 1),
                stride=(step_size, 1),
                padding=(window_size, 0),
            )
            waveform_batch = waveform_batch.permute(2, 0, 1)

        return waveform_batch

    @staticmethod
    def aggreagate(
        data: torch.Tensor,
        window_size: int,
        total_frames: int,
        step_size: Optional[int] = None,
        window="hanning",
    ):
        """
        stitch batched waveform into single waveform. (Overlap-add)
        arguments:
            data: batched waveform
            window_size : window_size used to batch waveform
            step_size : step_size used to batch waveform
            total_frames : total number of frames present in original waveform
            window : type of window used for overlap-add mechanism.
        """
        num_chunks, n_channels, num_frames = data.shape
        window = get_window(window=window, Nx=data.shape[-1])
        window = torch.from_numpy(window).to(data.device)
        data *= window

        data = data.permute(1, 2, 0)
        data = F.fold(
            data,
            (total_frames, 1),
            kernel_size=(window_size, 1),
            stride=(step_size, 1),
            padding=(window_size, 0),
        ).squeeze(-1)

        return data.reshape(1, n_channels, -1)

    @staticmethod
    def write_output(
        waveform: torch.Tensor, filename: Union[str, Path], sr: int
    ):
        """
        write audio output as wav file
        arguments:
            waveform : audio waveform
            filename : name of the wave file. Output will be written as cleaned_filename.wav
            sr : sampling rate
        """

        if isinstance(filename, str):
            filename = Path(filename)
        if filename.is_file():
            raise FileExistsError(f"file {filename} already exists")
        else:
            wavfile.write(filename, rate=sr, data=waveform.detach().cpu())

    @staticmethod
    def prepare_output(
        waveform: torch.Tensor,
        model_sampling_rate: int,
        audio: Union[str, np.ndarray, torch.Tensor],
        sampling_rate: Optional[int],
    ):
        """
        prepare output audio based on input format
        arguments:
            waveform : predicted audio waveform
            model_sampling_rate : sampling rate used to train the model
            audio : input audio
            sampling_rate : input audio sampling rate

        """
        if isinstance(audio, np.ndarray):
            waveform = waveform.detach().cpu().numpy()

        if sampling_rate is not None:
            waveform = Audio.resample_audio(
                waveform, sr=model_sampling_rate, target_sr=sampling_rate
            )

        return waveform
