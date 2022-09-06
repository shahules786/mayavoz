from torch import nn
import torch.nn.functional as F
import math 

from enhancer.utils.io import Audio as audio

class DeLSTM(nn.Module):
    def __init__(
        self,
        input_size:int,
        hidden_size:int,
        num_layers:int,
        bidirectional:bool=True

    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
        dim = 2 if bidirectional else 1
        self.linear = nn.Linear(dim*hidden_size,hidden_size)

    def forward(self,x):

        output,(h,c) = self.lstm(x)
        output = self.linear(output)

        return output

class Demucs(nn.Module):
    
    def __init__(
        self,
        c_in:int=1,
        c_out:int=1,
        hidden:int=48,
        kernel_size:int=8,
        stride:int=4,
        growth_factor:int=2,
        depth:int = 5,
        glu:bool = True,
        bidirectional:bool=True,
        resample:int=4,
        sampling_rate = 16000

    ):
        super().__init__()
        self.c_in = c_in 
        self.c_out = c_out 
        self.hidden = hidden
        self.growth_factor = growth_factor
        self.stride = stride
        self.kernel_size = kernel_size
        self.depth = depth
        self.bidirectional = bidirectional
        self.activation = nn.GLU(1) if glu else nn.ReLU()
        self.resample = resample
        self.sampling_rate = sampling_rate
        multi_factor = 2 if glu else 1

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for layer in range(self.depth):

            encoder_layer = [nn.Conv1d(c_in,hidden,kernel_size,stride),
                            nn.ReLU(),
                            nn.Conv1d(hidden, hidden*multi_factor,kernel_size,1),
                            self.activation]
            encoder_layer = nn.Sequential(*encoder_layer)
            self.encoder.append(encoder_layer)

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

    def forward(self,mixed_signal):
        
        length = mixed_signal.shape[-1]
        x = F.pad(mixed_signal, (0,self.get_padding_length(length) - length)) 
        if self.resample>1:
            x = audio.pt_resample_audio(audio=x, sr=self.sampling_rate,
                        target_sr=int(self.sampling_rate * self.resample))
        print("resampled->",x.shape)
        encoder_outputs = []
        for encoder in self.encoder:
            x = encoder(x)
            print(x.shape)
            encoder_outputs.append(x)
        x = x.permute(0,2,1)
        x = self.de_lstm(x)

        x = x.permute(0,2,1)
        for decoder in self.decoder:
            skip_connection = encoder_outputs.pop(-1)
            x += skip_connection[..., :x.shape[-1]]
            x = decoder(x)
        
        if self.resample > 1:
            x = audio.pt_resample_audio(x,int(self.sampling_rate * self.resample),
                                    self.sampling_rate)

        return x
        
    def get_padding_length(self,input_length):

        input_length = math.ceil(input_length * self.resample)

  
        for layer in range(self.depth):                                        # encoder operation
            input_length = math.ceil((input_length - self.kernel_size)/self.stride)+1
            input_length = max(1,input_length)
        for layer in range(self.depth):                                        # decoder operaration
            input_length = (input_length-1) * self.stride + self.kernel_size
        input_length = math.ceil(input_length/self.resample)

        return int(input_length)

        

        








        