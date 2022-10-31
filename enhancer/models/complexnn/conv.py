from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def init_weights(nnet):
    nn.init.xavier_normal_(nnet.weight.data)
    nn.init.constant_(nnet.bias, 0.0)
    return nnet


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
        dilation: int = 1,
    ):
        """
        Complex Conv2d (non-causal)
        """
        super().__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation

        self.real_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(self.padding[0], 0),
            groups=self.groups,
            dilation=self.dilation,
        )
        self.imag_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(self.padding[0], 0),
            groups=self.groups,
            dilation=self.dilation,
        )
        self.imag_conv = init_weights(self.imag_conv)
        self.real_conv = init_weights(self.real_conv)

    def forward(self, input):
        """
        complex axis should be always 1 dim
        """
        input = F.pad(input, [self.padding[1], self.padding[1], 0, 0])

        real, imag = torch.chunk(input, 2, 1)
        real_real = self.real_conv(real)
        real_imag = self.imag_conv(real)

        imag_imag = self.imag_conv(imag)
        imag_real = self.real_conv(imag)

        real = real_real - imag_imag
        imag = real_imag - imag_real

        out = torch.cat([real, imag], 1)

        return out


class ComplexConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        output_padding: Tuple[int, int] = (0, 0),
        groups: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels // 2
        self.out_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.output_padding = output_padding

        self.real_conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
        )

        self.imag_conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
        )

        init_weights(self.real_conv)
        init_weights(self.imag_conv)

    def forward(self, input):

        real, imag = torch.chunk(input, 2, 1)

        real_real = self.real_conv(real)
        real_imag = self.imag_conv(real)

        imag_imag = self.imag_conv(imag)
        imag_real = self.real_conv(imag)

        real = real_real - imag_imag
        imag = real_imag - imag_real

        out = torch.cat([real, imag], 1)

        return out
