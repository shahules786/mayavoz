from typing import List, Optional

import torch
from torch import nn


class ComplexLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        projection_size: Optional[int] = None,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size // 2
        self.hidden_size = hidden_size // 2
        self.num_layers = num_layers

        self.real_lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            bidirectional=bidirectional,
            batch_first=False,
        )
        self.imag_lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            bidirectional=bidirectional,
            batch_first=False,
        )

        bidirectional = 2 if bidirectional else 1
        if projection_size is not None:
            self.projection_size = projection_size // 2
            self.real_linear = nn.Linear(
                self.hidden_size * bidirectional, self.projection_size
            )
            self.imag_linear = nn.Linear(
                self.hidden_size * bidirectional, self.projection_size
            )
        else:
            self.projection_size = None

    def forward(self, input):

        if isinstance(input, List):
            real, imag = input
        else:
            real, imag = torch.chunk(input, 2, 1)

        real_real = self.real_lstm(real)[0]
        real_imag = self.imag_lstm(real)[0]

        imag_imag = self.imag_lstm(imag)[0]
        imag_real = self.real_lstm(imag)[0]

        real = real_real - imag_imag
        imag = imag_real + real_imag

        if self.projection_size is not None:
            real = self.real_linear(real)
            imag = self.imag_linear(imag)

        return [real, imag]
