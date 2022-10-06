import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from enhancer.data.dataset import EnhancerDataset
from enhancer.models.model import Model


class WavenetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        padding: int = 2,
        stride: int = 1,
        dilation: int = 1,
    ):
        super(WavenetDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, waveform):

        return self.decoder(waveform)


class WavenetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        padding: int = 7,
        stride: int = 1,
        dilation: int = 1,
    ):
        super(WavenetEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, waveform):
        return self.encoder(waveform)


class WaveUnet(Model):
    """
    Wave-U-Net model from  https://arxiv.org/pdf/1811.11307.pdf
    parameters:
        num_channels: int, defaults to 1
            number of channels in input audio
        depth : int, defaults to 12
            depth of network
        initial_output_channels: int, defaults to 24
            number of output channels in intial upsampling layer
        sampling_rate: int, defaults to 16KHz
            sampling rate of input audio
        lr : float, defaults to 1e-3
            learning rate used for training
        dataset: EnhancerDataset, optional
            EnhancerDataset object containing train/validation data for training
        duration : float, optional
            chunk duration in seconds
        loss : string or List of strings
            loss function to be used, available ("mse","mae","SI-SDR")
        metric : string or List of strings
            metric function to be used, available ("mse","mae","SI-SDR")
    """

    def __init__(
        self,
        num_channels: int = 1,
        depth: int = 12,
        initial_output_channels: int = 24,
        sampling_rate: int = 16000,
        lr: float = 1e-3,
        dataset: Optional[EnhancerDataset] = None,
        duration: Optional[float] = None,
        loss: Union[str, List] = "mse",
        metric: Union[str, List] = "mse",
    ):
        duration = (
            dataset.duration if isinstance(dataset, EnhancerDataset) else None
        )
        if dataset is not None:
            if sampling_rate != dataset.sampling_rate:
                logging.warning(
                    f"model sampling rate {sampling_rate} should match dataset sampling rate {dataset.sampling_rate}"
                )
                sampling_rate = dataset.sampling_rate
        super().__init__(
            num_channels=num_channels,
            sampling_rate=sampling_rate,
            lr=lr,
            dataset=dataset,
            duration=duration,
            loss=loss,
            metric=metric,
        )
        self.save_hyperparameters("depth")
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        out_channels = initial_output_channels
        for layer in range(depth):

            encoder = WavenetEncoder(num_channels, out_channels)
            self.encoders.append(encoder)

            num_channels = out_channels
            out_channels += initial_output_channels
            if layer == depth - 1:
                decoder = WavenetDecoder(
                    depth * initial_output_channels + num_channels, num_channels
                )
            else:
                decoder = WavenetDecoder(
                    num_channels + out_channels, num_channels
                )

            self.decoders.insert(0, decoder)

        bottleneck_dim = depth * initial_output_channels
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_dim, bottleneck_dim, 15, stride=1, padding=7),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv1d(1 + initial_output_channels, 1, kernel_size=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, waveform):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        if waveform.size(1) != 1:
            raise TypeError(
                f"Wave-U-Net can only process mono channel audio, input has {waveform.size(1)} channels"
            )

        encoder_outputs = []
        out = waveform
        for encoder in self.encoders:
            out = encoder(out)
            encoder_outputs.insert(0, out)
            out = out[:, :, ::2]

        out = self.bottleneck(out)

        for layer, decoder in enumerate(self.decoders):
            out = F.interpolate(out, scale_factor=2, mode="linear")
            out = self.fix_last_dim(out, encoder_outputs[layer])
            out = torch.cat([out, encoder_outputs[layer]], dim=1)
            out = decoder(out)

        out = torch.cat([out, waveform], dim=1)
        out = self.final(out)
        return out

    def fix_last_dim(self, x, target):
        """
        centre crop along last dimension
        """

        assert (
            x.shape[-1] >= target.shape[-1]
        ), "input dimension cannot be larger than target dimension"
        if x.shape[-1] == target.shape[-1]:
            return x

        diff = x.shape[-1] - target.shape[-1]
        if diff % 2 != 0:
            x = F.pad(x, (0, 1))
            diff += 1

        crop = diff // 2
        return x[:, :, crop:-crop]
