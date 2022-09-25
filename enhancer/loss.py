import torch
import torch.nn as nn


class mean_squared_error(nn.Module):

    def __init__(self,reduction="mean"):
        super().__init__()

        self.loss_fun = nn.MSELoss(reduction=reduction)

    def forward(self,prediction:torch.Tensor, target: torch.Tensor):

        if prediction.size() != target.size() or target.ndim < 3:
            raise TypeError(f"""Inputs must be of the same shape (batch_size,channels,samples) 
                            got {prediction.size()} and {target.size()} instead""")

        return self.loss_fun(prediction, target)

class mean_absolute_error(nn.Module):

    def __init__(self,reduction="mean"):
        super().__init__()

        self.loss_fun = nn.L1Loss(reduction=reduction)

    def forward(self, prediction:torch.Tensor, target: torch.Tensor):

        if prediction.size() != target.size() or target.ndim < 3:
            raise TypeError(f"""Inputs must be of the same shape (batch_size,channels,samples) 
                            got {prediction.size()} and {target.size()} instead""")

        return self.loss_fun(prediction, target)

class Avergeloss(nn.Module):

    def __init__(self,losses):
        super().__init__()

        self.valid_losses = nn.ModuleList()
        for loss in losses:
            loss = self.validate_loss(loss)
            self.valid_losses.append(loss())


    def validate_loss(self,loss:str):
        if loss not in LOSS_MAP.keys():
            raise ValueError(f"Invalid loss function {loss}, available loss functions are {LOSS_MAP.keys()}")
        else:
            return LOSS_MAP[loss]

    def forward(self,prediction:torch.Tensor, target:torch.Tensor):
        loss = 0.0
        for loss_fun in self.valid_losses:
            loss += loss_fun(prediction, target)
        
        return loss

            


LOSS_MAP = {"mea":mean_absolute_error, "mse": mean_squared_error}


