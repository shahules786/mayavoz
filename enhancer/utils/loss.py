import torch
import torch.nn as nn


class mean_squared_error(nn.Module):

    def __init__(self,reduction="mean"):
        super().__init__()

        self.loss_fun = nn.MSELoss(reduction=reduction)

    def forward(self,prediction:torch.Tensor, target: torch.Tensor):

        return self.loss_fun(prediction, target)

class mean_absolute_error(nn.Module):

    def __init__(self,reduction="mean"):

        self.loss_fun = nn.L1Loss(reduction=reduction)

    def forward(self, prediction:torch.Tensor, target: torch.Tensor):

        return self.loss_fun(prediction, target)
        
LOSS_MAP = {"mea":mean_absolute_error, "mse": mean_squared_error}


