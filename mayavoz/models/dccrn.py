import logging
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from mayavoz.data import EnhancerDataset
from mayavoz.models import Mayamodel
from mayavoz.models.complexnn import (
    ComplexBatchNorm2D,
    ComplexConv2d,
    ComplexConvTranspose2d,
    ComplexLSTM,
    ComplexRelu,
)
from mayavoz.models.complexnn.utils import complex_cat
from mayavoz.utils.transforms import ConviSTFT, ConvSTFT
from mayavoz.utils.utils import merge_dict


class DCCRN_ENCODER(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channel: int,
        kernel_size: Tuple[int, int],
        complex_norm: bool = True,
        complex_relu: bool = True,
        stride: Tuple[int, int] = (2, 1),
        padding: Tuple[int, int] = (2, 1),
    ):
        super().__init__()
        batchnorm = ComplexBatchNorm2D if complex_norm else nn.BatchNorm2d
        activation = ComplexRelu() if complex_relu else nn.PReLU()

        self.encoder = nn.Sequential(
            ComplexConv2d(
                in_channels,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            batchnorm(out_channel),
            activation,
        )

    def forward(self, waveform):

        return self.encoder(waveform)


class DCCRN_DECODER(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        layer: int = 0,
        complex_norm: bool = True,
        complex_relu: bool = True,
        stride: Tuple[int, int] = (2, 1),
        padding: Tuple[int, int] = (2, 0),
        output_padding: Tuple[int, int] = (1, 0),
    ):
        super().__init__()
        batchnorm = ComplexBatchNorm2D if complex_norm else nn.BatchNorm2d
        activation = ComplexRelu() if complex_relu else nn.PReLU()

        if layer != 0:
            self.decoder = nn.Sequential(
                ComplexConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                ),
                batchnorm(out_channels),
                activation,
            )
        else:
            self.decoder = nn.Sequential(
                ComplexConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            )

    def forward(self, waveform):

        return self.decoder(waveform)


class DCCRN(Mayamodel):

    STFT_DEFAULTS = {
        "window_len": 400,
        "hop_size": 100,
        "nfft": 512,
        "window": "hamming",
    }

    ED_DEFAULTS = {
        "initial_output_channels": 32,
        "depth": 6,
        "kernel_size": 5,
        "growth_factor": 2,
        "stride": 2,
        "padding": 2,
        "output_padding": 1,
    }

    LSTM_DEFAULTS = {
        "num_layers": 2,
        "hidden_size": 256,
    }

    def __init__(
        self,
        stft: Optional[dict] = None,
        encoder_decoder: Optional[dict] = None,
        lstm: Optional[dict] = None,
        complex_lstm: bool = True,
        complex_norm: bool = True,
        complex_relu: bool = True,
        masking_mode: str = "E",
        num_channels: int = 1,
        sampling_rate=16000,
        lr: float = 1e-3,
        dataset: Optional[EnhancerDataset] = None,
        duration: Optional[float] = None,
        loss: Union[str, List, Any] = "mse",
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

        encoder_decoder = merge_dict(self.ED_DEFAULTS, encoder_decoder)
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        stft = merge_dict(self.STFT_DEFAULTS, stft)
        self.save_hyperparameters(
            "encoder_decoder",
            "lstm",
            "stft",
            "complex_lstm",
            "complex_norm",
            "masking_mode",
        )
        self.complex_lstm = complex_lstm
        self.complex_norm = complex_norm
        self.masking_mode = masking_mode

        self.stft = ConvSTFT(
            stft["window_len"], stft["hop_size"], stft["nfft"], stft["window"]
        )
        self.istft = ConviSTFT(
            stft["window_len"], stft["hop_size"], stft["nfft"], stft["window"]
        )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        num_channels *= 2
        hidden_size = encoder_decoder["initial_output_channels"]
        growth_factor = 2

        for layer in range(encoder_decoder["depth"]):

            encoder_ = DCCRN_ENCODER(
                num_channels,
                hidden_size,
                kernel_size=(encoder_decoder["kernel_size"], 2),
                stride=(encoder_decoder["stride"], 1),
                padding=(encoder_decoder["padding"], 1),
                complex_norm=complex_norm,
                complex_relu=complex_relu,
            )
            self.encoder.append(encoder_)

            decoder_ = DCCRN_DECODER(
                hidden_size + hidden_size,
                num_channels,
                layer=layer,
                kernel_size=(encoder_decoder["kernel_size"], 2),
                stride=(encoder_decoder["stride"], 1),
                padding=(encoder_decoder["padding"], 0),
                output_padding=(encoder_decoder["output_padding"], 0),
                complex_norm=complex_norm,
                complex_relu=complex_relu,
            )

            self.decoder.insert(0, decoder_)

            if layer < encoder_decoder["depth"] - 3:
                num_channels = hidden_size
                hidden_size *= growth_factor
            else:
                num_channels = hidden_size

        kernel_size = hidden_size / 2
        hidden_size = stft["nfft"] / 2 ** (encoder_decoder["depth"])

        if self.complex_lstm:
            lstms = []
            for layer in range(lstm["num_layers"]):

                if layer == 0:
                    input_size = int(hidden_size * kernel_size)
                else:
                    input_size = lstm["hidden_size"]

                if layer == lstm["num_layers"] - 1:
                    projection_size = int(hidden_size * kernel_size)
                else:
                    projection_size = None

                kwargs = {
                    "input_size": input_size,
                    "hidden_size": lstm["hidden_size"],
                    "num_layers": 1,
                }

                lstms.append(
                    ComplexLSTM(projection_size=projection_size, **kwargs)
                )
            self.lstm = nn.Sequential(*lstms)
        else:
            self.lstm = nn.Sequential(
                nn.LSTM(
                    input_size=hidden_size * kernel_size,
                    hidden_sizs=lstm["hidden_size"],
                    num_layers=lstm["num_layers"],
                    dropout=0.0,
                    batch_first=False,
                )[0],
                nn.Linear(lstm["hidden"], hidden_size * kernel_size),
            )

    def forward(self, waveform):

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        if waveform.size(1) != self.hparams.num_channels:
            raise ValueError(
                f"Number of input channels initialized is {self.hparams.num_channels} but got {waveform.size(1)} channels"
            )

        waveform_stft = self.stft(waveform)
        real = waveform_stft[:, : self.stft.nfft // 2 + 1]
        imag = waveform_stft[:, self.stft.nfft // 2 + 1 :]

        mag_spec = torch.sqrt(real**2 + imag**2 + 1e-9)
        phase_spec = torch.atan2(imag, real)
        complex_spec = torch.stack([mag_spec, phase_spec], 1)[:, :, 1:]

        encoder_outputs = []
        out = complex_spec
        for _, encoder in enumerate(self.encoder):
            out = encoder(out)
            encoder_outputs.append(out)

        B, C, D, T = out.size()
        out = out.permute(3, 0, 1, 2)
        if self.complex_lstm:

            lstm_real = out[:, :, : C // 2]
            lstm_imag = out[:, :, C // 2 :]
            lstm_real = lstm_real.reshape(T, B, C // 2 * D)
            lstm_imag = lstm_imag.reshape(T, B, C // 2 * D)
            lstm_real, lstm_imag = self.lstm([lstm_real, lstm_imag])
            lstm_real = lstm_real.reshape(T, B, C // 2, D)
            lstm_imag = lstm_imag.reshape(T, B, C // 2, D)
            out = torch.cat([lstm_real, lstm_imag], 2)
        else:
            out = out.reshape(T, B, C * D)
            out = self.lstm(out)
            out = out.reshape(T, B, D, C)

        out = out.permute(1, 2, 3, 0)
        for layer, decoder in enumerate(self.decoder):
            skip_connection = encoder_outputs.pop(-1)
            out = complex_cat([skip_connection, out])
            out = decoder(out)
            out = out[..., 1:]
        mask_real, mask_imag = out[:, 0], out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])
        if self.masking_mode == "E":

            mask_mag = torch.sqrt(mask_real**2 + mask_imag**2)
            real_phase = mask_real / (mask_mag + 1e-8)
            imag_phase = mask_imag / (mask_mag + 1e-8)
            mask_phase = torch.atan2(imag_phase, real_phase)
            mask_mag = torch.tanh(mask_mag)
            est_mag = mask_mag * mag_spec
            est_phase = mask_phase * phase_spec
            # cos(theta) + isin(theta)
            real = est_mag + torch.cos(est_phase)
            imag = est_mag + torch.sin(est_phase)

        if self.masking_mode == "C":

            real = real * mask_real - imag * mask_imag
            imag = real * mask_imag + imag * mask_real

        else:

            real = real * mask_real
            imag = imag * mask_imag

        spec = torch.cat([real, imag], 1)
        wav = self.istft(spec)
        wav = wav.clamp_(-1, 1)
        return wav
