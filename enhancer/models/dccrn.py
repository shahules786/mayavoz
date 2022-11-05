import logging
from typing import Any, List, Optional, Tuple, Union

from torch import nn

from enhancer.data import EnhancerDataset
from enhancer.models import Model
from enhancer.models.complexnn import ComplexConv2d, ComplexLSTM
from enhancer.models.complexnn.conv import ComplexConvTranspose2d
from enhancer.models.complexnn.utils import ComplexBatchNorm2D, ComplexRelu
from enhancer.utils.transforms import ConviSTFT, ConvSTFT
from enhancer.utils.utils import merge_dict


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
        complex_norm: bool = True,
        complex_relu: bool = True,
        stride: Tuple[int, int] = (2, 1),
        padding: Tuple[int, int] = (2, 0),
        output_padding: Tuple[int, int] = (1, 0),
    ):
        super().__init__()
        batchnorm = ComplexBatchNorm2D if complex_norm else nn.BatchNorm2d
        activation = ComplexRelu() if complex_relu else nn.PReLU()

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

    def forward(self, waveform):

        return self.decoder(waveform)


class DCCRN(Model):

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
            self.lstm = nn.LSTM(
                input_size=hidden_size * kernel_size,
                hidden_sizs=lstm["hidden_size"],
                num_layers=lstm["num_layers"],
                dropout=0.0,
                batch_first=False,
            )

    def forward(self, waveform):

        return waveform
