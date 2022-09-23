from turtle import forward
import torch.nn as nn


class WavenetDecoder(nn.Module):

    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        padding:int,
        stride:int,
        dilation:int=1,
    ):
        super(WavenetDecoder,self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
    
    def forward(self,waveform):
        
        return self.decoder(waveform)

class WavenetEncoder(nn.Module):

    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        padding:int,
        stride:int,
        dilation:int=1,
    ):
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)  
        )
        

    def forward(
        self,
        waveform
    ):
        return self.encoder(waveform)




class WaveUnet(nn.Module):

    def __init__(
        self
    ):
        pass

    def forward(
        self,waveform
    ):
        pass