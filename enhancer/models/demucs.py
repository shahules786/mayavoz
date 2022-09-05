from typing import bool
from torch import nn 

class DeLSTM(nn.Module):
    def __init__(
        self,
        input_size:int,
        hidden_size:int,
        num_layers:int,
        bidirectional:bool=True

    ):
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
        dim = 2 if bidirectional else 1
        self.linear = nn.Linear(dim*hidden_size,hidden_size)

    def forward(self,x):

        output,(h,c) = self.lstm(x)
        output = self.linear(output)

        return output

class Demus(nn.Module):
    
    def __init__(
        self,
        c_in:int=1,
        c_out:int=1,
        hidden:int=48,
        kernel_size:int=8,
        stride:int=4,
        growth_factor:int=2,
        depth:int = 6,
        glu:bool = True,
        bidirectional:bool=True,
        resample:int=2,

    ):
        self.c_in = c_in 
        self.c_out = c_out 
        self.hidden = hidden
        self.growth_factor = growth_factor
        self.stride = stride
        self.kernel_size = kernel_size
        self.depth = depth
        self.bidirectional = bidirectional
        self.activation = nn.GLU(1) if glu else nn.ReLU()
        multi_factor = 2 if glu else 1

        ## do resampling

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for layer in range(self.depth):

            encoder_layer = [nn.Conv1d(c_in,hidden,kernel_size,stride),
                            nn.ReLU(),
                            nn.Conv1d(hidden, hidden*multi_factor,kernel_size,1),
                            self.activation]
            encoder_layer = nn.Sequential(encoder_layer)
            self.encoder.append(*encoder_layer)

            decoder_layer = [nn.Conv1d(hidden,hidden*multi_factor,kernel_size,1),
                            self.activation,
                            nn.ConvTranspose1d(hidden,c_out,kernel_size,stride)
                            ]
            if layer>0:
                decoder_layer.append(nn.ReLU())
            decoder_layer = nn.Sequential(*decoder_layer)
            self.decoder.insert(0,decoder_layer)

            c_out = hidden
            c_in = hidden
            hidden = self.growth_factor * hidden

        
        self.de_lstm = DeLSTM(input_size=c_in,hidden_size=c_in,num_layers=2,bidirectional=self.bidirectional)

    def forward(self,input):
        








        