import torch
from torch import nn


class ComplexBatchNorm2D(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        """
        Complex batch normalization 2D
        https://arxiv.org/abs/1705.09792


        """
        super().__init__()
        self.num_features = num_features // 2
        self.affine = affine
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        self.eps = eps

        if self.affine:
            self.Wrr = nn.parameter.Parameter(torch.Tensor(self.num_features))
            self.Wri = nn.parameter.Parameter(torch.Tensor(self.num_features))
            self.Wii = nn.parameter.Parameter(torch.Tensor(self.num_features))
            self.Br = nn.parameter.Parameter(torch.Tensor(self.num_features))
            self.Bi = nn.parameter.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter("Wrr", None)
            self.register_parameter("Wri", None)
            self.register_parameter("Wii", None)
            self.register_parameter("Br", None)
            self.register_parameter("Bi", None)

        if self.track_running_stats:
            values = torch.zeros(self.num_features)
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
            self.Wii.data.fill_(1)
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

    def extra_repr(self):
        return "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}".format(
            **self.__dict__
        )

    def forward(self, input):

        real, imag = torch.chunk(input, 2, 1)
        exp_avg_factor = 0.0

        training = self.training and self.track_running_stats
        if training:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exp_avg_factor = 1 / self.num_batches_tracked
            else:
                exp_avg_factor = self.momentum

        redux = [i for i in reversed(range(real.dim())) if i != 1]
        vdim = [1] * real.dim()
        vdim[1] = real.size(1)

        if training:
            batch_mean_real, batch_mean_imag = real, imag
            for dim in redux:
                batch_mean_real = batch_mean_real.mean(dim, keepdim=True)
                batch_mean_imag = batch_mean_imag.mean(dim, keepdim=True)
            if self.track_running_stats:
                self.Mean_real.lerp_(batch_mean_real.squeeze(), exp_avg_factor)
                self.Mean_imag.lerp_(batch_mean_imag.squeeze(), exp_avg_factor)

        else:
            batch_mean_real = self.Mean_real.view(vdim)
            batch_mean_imag = self.Mean_imag.view(vdim)

        real = real - batch_mean_real
        imag = imag - batch_mean_imag

        if training:
            batch_var_rr = real * real
            batch_var_ri = real * imag
            batch_var_ii = imag * imag
            for dim in redux:
                batch_var_rr = batch_var_rr.mean(dim, keepdim=True)
                batch_var_ri = batch_var_ri.mean(dim, keepdim=True)
                batch_var_ii = batch_var_ii.mean(dim, keepdim=True)
            if self.track_running_stats:
                self.Var_rr.lerp_(batch_var_rr.squeeze(), exp_avg_factor)
                self.Var_ri.lerp_(batch_var_ri.squeeze(), exp_avg_factor)
                self.Var_ii.lerp_(batch_var_ii.squeeze(), exp_avg_factor)
        else:
            batch_var_rr = self.Var_rr.view(vdim)
            batch_var_ii = self.Var_ii.view(vdim)
            batch_var_ri = self.Var_ri.view(vdim)

        batch_var_rr += self.eps
        batch_var_ii += self.eps

        # Covariance matrics
        # | batch_var_rr    batch_var_ri |
        # | batch_var_ir    batch_var_ii |  here batch_var_ir == batch_var_ri
        # Inverse square root of cov matrix by combining below two formulas
        # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        # https://mathworld.wolfram.com/MatrixInverse.html

        tau = batch_var_rr + batch_var_ii
        s = batch_var_rr * batch_var_ii - batch_var_ri * batch_var_ri
        t = (tau + 2 * s).sqrt()

        rst = (s * t).reciprocal()
        Urr = (batch_var_ii + s) * rst
        Uri = -batch_var_ri * rst
        Uii = (batch_var_rr + s) * rst

        if self.affine:
            Wrr, Wri, Wii = (
                self.Wrr.view(vdim),
                self.Wri.view(vdim),
                self.Wii.view(vdim),
            )
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wii * Uri) + (Wri * Urr)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * real) + (Zri * imag)
        yi = (Zir * real) + (Zii * imag)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        outputs = torch.cat([yr, yi], 1)
        return outputs


class ComplexRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.real_relu = nn.PReLU()
        self.imag_relu = nn.PReLU()

    def forward(self, input):

        real, imag = torch.chunk(input, 2, 1)
        real = self.real_relu(real)
        imag = self.imag_relu(imag)
        return torch.cat([real, imag], dim=1)


def complex_cat(inputs, axis=1):

    real, imag = [], []
    for data in inputs:
        real_data, imag_data = torch.chunk(data, 2, axis)
        real.append(real_data)
        imag.append(imag_data)
    real = torch.cat(real, axis)
    imag = torch.cat(imag, axis)
    return torch.cat([real, imag], axis)
