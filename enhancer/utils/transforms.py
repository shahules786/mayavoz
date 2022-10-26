from typing import Optional

import numpy as np
import torch
from scipy.signal import get_window
from torch import nn


class ConvFFT(nn.Module):
    def __init__(
        self,
        window_len: int,
        nfft: Optional[int] = None,
        window: str = "hamming",
    ):
        self.window_len = window_len
        self.nfft = nfft if nfft else np.int(2 ** np.ceil(np.log2(window_len)))
        self.window = get_window(window, window_len)

    @property
    def init_kernel(self):

        fourier_basis = np.fft.rfft(np.eye(self.nfft))[: self.window_len]
        real, imag = np.real(fourier_basis), np.imag(fourier_basis)
        kernel = np.concatenate([real, imag], 1).T
        kernel *= self.window
        return torch.from_numpy(kernel)


class ConvSTFT(ConvFFT):
    def __init__(
        self,
        window_len: int,
        hop_size: Optional[int] = None,
        nfft: Optional[int] = None,
        window: str = "hamming",
    ):
        super(self, ConvSTFT).__init__(
            window_len=window_len, nfft=nfft, window=window
        )
        self.hop_size = hop_size if hop_size else window_len // 2
        self.register_buffer("weight", self.init_kernel)

    def forward(self, input):
        pass


class ConviSTFT(ConvFFT):
    def __init__(
        self,
        window_len: int,
        hop_size: Optional[int] = None,
        nfft: Optional[int] = None,
        window: str = "hamming",
    ):
        super(self, ConvSTFT).__init__(
            window_len=window_len, nfft=nfft, window=window
        )
        self.hop_size = hop_size if hop_size else window_len // 2
        self.register_buffer("weight", self.init_kernel)

    def forward(self, input):
        pass
