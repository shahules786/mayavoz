import torch
from torch import nn


class ComplexBatchNorm(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: bool = True,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        self.num_features = num_features // 2
        self.affine = affine
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        if self.affine:
            values = torch.Tensor(self.num_features)
            self.Wrr = nn.parameter.Parameter(values)
            self.Wri = nn.parameter.Parameter(values)
            self.Wii = nn.parameter.Parameter(values)
            self.Br = nn.parameter.Parameter(values)
            self.Bi = nn.parameter.Parameter(values)
        else:
            self.register_parameter("Wrr", None)
            self.register_parameter("Wri", None)
            self.register_parameter("Wii", None)
            self.register_parameter("Br", None)
            self.register_parameter("Bi", None)

        if self.track_running_stats:
            values = torch.Tensor(self.num_features)
            self.register_buffer("Mean_real", values)
            self.register_buffer("Mean_imag", values)
            self.register_buffer("Var_rr", values)
            self.register_buffer("Var_ri", values)
            self.register_buffer("Var_ii", values)
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("Mean_real", None)
            self.register_parameter("Mean_imag", None)
            self.register_parameter("Var_rr", None)
            self.register_parameter("Var_ri", None)
            self.register_parameter("Var_ii", None)
            self.register_parameter("num_batches_tracked", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.Wrr.data.fill_(1)
            self.Wii.data.fill(1)
            self.Wri.data.uniform_(-0.9, 0.9)
            self.Br.data.fill_(0)
            self.Bi.data.fill_(0)
        self.reset_running_stats()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.Mean_real.zero_()
            self.Mean_imag.zero_()
            self.Var_rr.fill_(1)
            self.Var_ri.zero_()
            self.Var_ii.fill_(1)
            self.num_batches_tracked.zero_()

    def forward(self, input):
        pass
