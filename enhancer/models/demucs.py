import logging
import math
from typing import List, Optional, Union

import torch.nn.functional as F
from torch import nn

from enhancer.data.dataset import EnhancerDataset
from enhancer.models.model import Model
from enhancer.utils.io import Audio as audio
from enhancer.utils.utils import merge_dict


class DemucsLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bidirectional=bidirectional
        )
        dim = 2 if bidirectional else 1
        self.linear = nn.Linear(dim * hidden_size, hidden_size)

    def forward(self, x):

        output, (h, c) = self.lstm(x)
        output = self.linear(output)

        return output, (h, c)


class DemucsEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        hidden_size: int,
        kernel_size: int,
        stride: int = 1,
        glu: bool = False,
    ):
        super().__init__()
        activation = nn.GLU(1) if glu else nn.ReLU()
        multi_factor = 2 if glu else 1
        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, hidden_size, kernel_size, stride),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size * multi_factor, 1, 1),
            activation,
        )

    def forward(self, waveform):

        return self.encoder(waveform)


class DemucsDecoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        hidden_size: int,
        kernel_size: int,
        stride: int = 1,
        glu: bool = False,
        layer: int = 0,
    ):
        super().__init__()
        activation = nn.GLU(1) if glu else nn.ReLU()
        multi_factor = 2 if glu else 1
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size * multi_factor, 1, 1),
            activation,
            nn.ConvTranspose1d(hidden_size, num_channels, kernel_size, stride),
        )
        if layer > 0:
            self.decoder.add_module("4", nn.ReLU())

    def forward(
        self,
        waveform,
    ):

        out = self.decoder(waveform)
        return out


class Demucs(Model):
    """
    Demucs model from https://arxiv.org/pdf/1911.13254.pdf
    parameters:
        encoder_decoder: dict, optional
            keyword arguments passsed to encoder decoder block
        lstm : dict, optional
            keyword arguments passsed to LSTM block
        num_channels: int, defaults to 1
            number channels in input audio
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

    ED_DEFAULTS = {
        "initial_output_channels": 48,
        "kernel_size": 8,
        "stride": 4,
        "depth": 5,
        "glu": True,
        "growth_factor": 2,
    }
    LSTM_DEFAULTS = {
        "bidirectional": True,
        "num_layers": 2,
    }

    def __init__(
        self,
        encoder_decoder: Optional[dict] = None,
        lstm: Optional[dict] = None,
        num_channels: int = 1,
        resample: int = 4,
        sampling_rate=16000,
        lr: float = 1e-3,
        dataset: Optional[EnhancerDataset] = None,
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

        encoder_decoder = merge_dict(self.ED_DEFAULTS, encoder_decoder)
        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        self.save_hyperparameters("encoder_decoder", "lstm", "resample")
        hidden = encoder_decoder["initial_output_channels"]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for layer in range(encoder_decoder["depth"]):

            encoder_layer = DemucsEncoder(
                num_channels=num_channels,
                hidden_size=hidden,
                kernel_size=encoder_decoder["kernel_size"],
                stride=encoder_decoder["stride"],
                glu=encoder_decoder["glu"],
            )
            self.encoder.append(encoder_layer)

            decoder_layer = DemucsDecoder(
                num_channels=num_channels,
                hidden_size=hidden,
                kernel_size=encoder_decoder["kernel_size"],
                stride=encoder_decoder["stride"],
                glu=encoder_decoder["glu"],
                layer=layer,
            )
            self.decoder.insert(0, decoder_layer)

            num_channels = hidden
            hidden = self.ED_DEFAULTS["growth_factor"] * hidden

        self.de_lstm = DemucsLSTM(
            input_size=num_channels,
            hidden_size=num_channels,
            num_layers=lstm["num_layers"],
            bidirectional=lstm["bidirectional"],
        )

    def forward(self, waveform):

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        if waveform.size(1) != 1:
            raise TypeError(
                f"Demucs can only process mono channel audio, input has {waveform.size(1)} channels"
            )

        length = waveform.shape[-1]
        x = F.pad(waveform, (0, self.get_padding_length(length) - length))
        if self.hparams.resample > 1:
            x = audio.resample_audio(
                audio=x,
                sr=self.hparams.sampling_rate,
                target_sr=int(
                    self.hparams.sampling_rate * self.hparams.resample
                ),
            )

        encoder_outputs = []
        for encoder in self.encoder:
            x = encoder(x)
            encoder_outputs.append(x)
        x = x.permute(0, 2, 1)
        x, _ = self.de_lstm(x)

        x = x.permute(0, 2, 1)
        for decoder in self.decoder:
            skip_connection = encoder_outputs.pop(-1)
            x += skip_connection[..., : x.shape[-1]]
            x = decoder(x)

        if self.hparams.resample > 1:
            x = audio.resample_audio(
                x,
                int(self.hparams.sampling_rate * self.hparams.resample),
                self.hparams.sampling_rate,
            )

        return x[..., :length]

    def get_padding_length(self, input_length):

        input_length = math.ceil(input_length * self.hparams.resample)

        for layer in range(
            self.hparams.encoder_decoder["depth"]
        ):  # encoder operation
            input_length = (
                math.ceil(
                    (input_length - self.hparams.encoder_decoder["kernel_size"])
                    / self.hparams.encoder_decoder["stride"]
                )
                + 1
            )
            input_length = max(1, input_length)
        for layer in range(
            self.hparams.encoder_decoder["depth"]
        ):  # decoder operaration
            input_length = (input_length - 1) * self.hparams.encoder_decoder[
                "stride"
            ] + self.hparams.encoder_decoder["kernel_size"]
        input_length = math.ceil(input_length / self.hparams.resample)

        return int(input_length)
