import torch
import torch.nn as nn


class mean_squared_error(nn.Module):

    def __init__(self,reduction="mean"):
        super().__init__()

        self.loss_fun = nn.MSELoss(reduction=reduction)
        self.higher_better = False

    def forward(self,prediction:torch.Tensor, target: torch.Tensor):

        if prediction.size() != target.size() or target.ndim < 3:
            raise TypeError(f"""Inputs must be of the same shape (batch_size,channels,samples) 
                            got {prediction.size()} and {target.size()} instead""")

        return self.loss_fun(prediction, target)

class mean_absolute_error(nn.Module):

    def __init__(self,reduction="mean"):
        super().__init__()

        self.loss_fun = nn.L1Loss(reduction=reduction)
        self.higher_better = False

    def forward(self, prediction:torch.Tensor, target: torch.Tensor):

        if prediction.size() != target.size() or target.ndim < 3:
            raise TypeError(f"""Inputs must be of the same shape (batch_size,channels,samples) 
                            got {prediction.size()} and {target.size()} instead""")

        return self.loss_fun(prediction, target)

class Si_SDR(nn.Module):

    def __init__(
        self,
        reduction:str="mean"
    ):
        super().__init__()
        if reduction in ["sum","mean",None]:
            self.reduction = reduction
        else:
            raise TypeError("Invalid reduction, valid options are sum, mean, None")
        self.higher_better = False

    def forward(self,prediction:torch.Tensor, target:torch.Tensor):

        if prediction.size() != target.size() or target.ndim < 3:
            raise TypeError(f"""Inputs must be of the same shape (batch_size,channels,samples) 
                            got {prediction.size()} and {target.size()} instead""")
        
        target_energy = torch.sum(target**2,keepdim=True,dim=-1)
        scaling_factor = torch.sum(prediction*target,keepdim=True,dim=-1) / target_energy
        target_projection = target * scaling_factor
        noise = prediction - target_projection
        ratio = torch.sum(target_projection**2,dim=-1) / torch.sum(noise**2,dim=-1)
        si_sdr = 10*torch.log10(ratio).mean(dim=-1)

        if self.reduction == "sum":
            si_sdr = si_sdr.sum()
        elif self.reduction == "mean":
            si_sdr = si_sdr.mean()
        else:
            pass
    
        return si_sdr



class Avergeloss(nn.Module):

    def __init__(self,losses):
        super().__init__()

        self.valid_losses = nn.ModuleList()
        
        direction = [getattr(LOSS_MAP[loss](),"higher_better") for loss in losses]
        if len(set(direction)) > 1:
            raise ValueError("all cost functions should be of same nature, maximize or minimize!")

        self.higher_better = direction[0]
        for loss in losses:
            loss = self.validate_loss(loss)
            self.valid_losses.append(loss())


    def validate_loss(self,loss:str):
        if loss not in LOSS_MAP.keys():
            raise ValueError(f"Invalid loss function {loss}, available loss functions are {tuple([loss for loss in LOSS_MAP.keys()])}")
        else:
            return LOSS_MAP[loss]

    def forward(self,prediction:torch.Tensor, target:torch.Tensor):
        loss = 0.0
        for loss_fun in self.valid_losses:
            loss += loss_fun(prediction, target)
        
        return loss

            


LOSS_MAP = {"mae":mean_absolute_error,
            "mse": mean_squared_error,
            "SI-SDR":Si_SDR}


