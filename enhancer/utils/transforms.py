from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import get_window
from torch import nn


class ConvFFT(nn.Module):
    def __init__(
        self,
        window_len: int,
        nfft: Optional[int] = None,
        window: str = "hamming",
    ):
        super().__init__()
        self.window_len = window_len
        self.nfft = nfft if nfft else np.int(2 ** np.ceil(np.log2(window_len)))
        self.window = torch.from_numpy(
            get_window(window, window_len, fftbins=True).astype("float32")
        )

    def init_kernel(self, inverse=False):

        fourier_basis = np.fft.rfft(np.eye(self.nfft))[: self.window_len]
        real, imag = np.real(fourier_basis), np.imag(fourier_basis)
        kernel = np.concatenate([real, imag], 1).T
        if inverse:
            kernel = np.linalg.pinv(kernel).T
        kernel = torch.from_numpy(kernel.astype("float32")).unsqueeze(1)
        kernel *= self.window
        return kernel


class ConvSTFT(ConvFFT):
    def __init__(
        self,
        window_len: int,
        hop_size: Optional[int] = None,
        nfft: Optional[int] = None,
        window: str = "hamming",
    ):
        super().__init__(window_len=window_len, nfft=nfft, window=window)
        self.hop_size = hop_size if hop_size else window_len // 2
        self.register_buffer("weight", self.init_kernel())

    def forward(self, input):

        if input.dim() < 2:
            raise ValueError(
                f"Expected signal with shape 2 or 3 got {input.dim()}"
            )
        elif input.dim() == 2:
            input = input.unsqueeze(1)
        else:
            pass
        input = F.pad(
            input,
            (self.window_len - self.hop_size, self.window_len - self.hop_size),
        )
        output = F.conv1d(input, self.weight, stride=self.hop_size)

        return output


class ConviSTFT(ConvFFT):
    def __init__(
        self,
        window_len: int,
        hop_size: Optional[int] = None,
        nfft: Optional[int] = None,
        window: str = "hamming",
    ):
        super().__init__(window_len=window_len, nfft=nfft, window=window)
        self.hop_size = hop_size if hop_size else window_len // 2
        self.register_buffer("weight", self.init_kernel(True))
        self.register_buffer("enframe", torch.eye(window_len).unsqueeze(1))

    def forward(self, input, phase=None):

        if phase is not None:
            real = input * torch.cos(phase)
            imag = input * torch.sin(phase)
            input = torch.cat([real, imag], 1)
        out = F.conv_transpose1d(input, self.weight, stride=self.hop_size)
        coeff = self.window.unsqueeze(1).repeat(1, 1, input.size(-1)) ** 2
        coeff = F.conv_transpose1d(coeff, self.enframe, stride=self.hop_size)
        out = out / (coeff + 1e-8)
        pad = self.window_len - self.hop_size
        out = out[..., pad:-pad]
        return out
