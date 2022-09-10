from turtle import forward
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

class Avergeloss(nn.Module):

    def __init__(self,losses):

        self.valid_losses = nn.ModuleList()
        for loss in losses:
            loss = self.validate_loss(loss)
            self.valid_losses.append(loss())


    def validate_loss(self,loss:str):
        if loss not in LOSS_MAP.keys():
            raise ValueError()
        else:
            return LOSS_MAP[loss]

    def forward(self,prediction:torch.Tensor, target:torch.Tensor):
        loss = 0.0
        for loss_fun in self.valid_losses:
            loss += loss_fun(prediction, target)
        
        return loss

            


LOSS_MAP = {"mea":mean_absolute_error, "mse": mean_squared_error}


