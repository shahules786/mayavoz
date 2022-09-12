from typing import Optional, Union, List
from torch import nn
import torch.nn.functional as F
import math 

from enhancer.models.model import Model
from enhancer.data.dataset import EnhancerDataset
from enhancer.utils.io import Audio as audio
from enhancer.utils.utils import merge_dict

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

        return output,(h,c)

class Demucs(Model):

    ED_DEFAULTS = {
        "initial_output_channels":48,
        "kernel_size":8,
        "stride":1,
        "depth":5,
        "glu":True,
        "growth_factor":2,
    }
    LSTM_DEFAULTS = {
        "bidirectional":True,
        "num_layers":2,
    }
    
    def __init__(
        self,
        encoder_decoder:Optional[dict]=None,
        lstm:Optional[dict]=None,
        num_channels:int=1,
        resample:int=4,
        sampling_rate = 16000,
        lr:float=1e-3,
        dataset:Optional[EnhancerDataset]=None,
        loss:Union[str, List] = "mse"

    ):
        super().__init__(num_channels=num_channels,
                            sampling_rate=sampling_rate,lr=lr,
                            dataset=dataset,loss=loss)
        
        encoder_decoder = merge_dict(self.ED_DEFAULTS,encoder_decoder)
        lstm = merge_dict(self.LSTM_DEFAULTS,lstm)
        self.save_hyperparameters("encoder_decoder","lstm","resample")
        
        hidden = encoder_decoder["initial_output_channels"]
        activation = nn.GLU(1) if encoder_decoder["glu"] else nn.ReLU()
        multi_factor = 2 if encoder_decoder["glu"] else 1

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for layer in range(encoder_decoder["depth"]):

            encoder_layer = [nn.Conv1d(num_channels,hidden,encoder_decoder["kernel_size"],encoder_decoder["stride"]),
                            nn.ReLU(),
                            nn.Conv1d(hidden, hidden*multi_factor,encoder_decoder["kernel_size"],1),
                            activation]
            encoder_layer = nn.Sequential(*encoder_layer)
            self.encoder.append(encoder_layer)

            decoder_layer = [nn.Conv1d(hidden,hidden*multi_factor,encoder_decoder["kernel_size"],1),
                            activation,
                            nn.ConvTranspose1d(hidden,num_channels,encoder_decoder["kernel_size"],encoder_decoder["stride"])
                            ]
            if layer>0:
                decoder_layer.append(nn.ReLU())
            decoder_layer = nn.Sequential(*decoder_layer)
            self.decoder.insert(0,decoder_layer)

            num_channels = hidden
            hidden = self.ED_DEFAULTS["growth_factor"] * hidden

        
        self.de_lstm = DeLSTM(input_size=num_channels,hidden_size=num_channels,num_layers=lstm["num_layers"],bidirectional=lstm["bidirectional"])

    def forward(self,mixed_signal):

        if mixed_signal.dim() == 2:
            mixed_signal = mixed_signal.unsqueeze(1)

        length = mixed_signal.shape[-1]
        x = F.pad(mixed_signal, (0,self.get_padding_length(length) - length)) 
        if self.hparams.resample>1:
            x = audio.pt_resample_audio(audio=x, sr=self.hparams.sampling_rate,
                        target_sr=int(self.hparams.sampling_rate * self.hparams.resample))
        
        encoder_outputs = []
        for encoder in self.encoder:
            x = encoder(x)
            encoder_outputs.append(x)
        x = x.permute(0,2,1)
        x,_ = self.de_lstm(x)

        x = x.permute(0,2,1)
        for decoder in self.decoder:
            skip_connection = encoder_outputs.pop(-1)
            x += skip_connection[..., :x.shape[-1]]
            x = decoder(x)
        
        if self.hparams.resample > 1:
            x = audio.pt_resample_audio(x,int(self.hparams.sampling_rate * self.hparams.resample),
                                    self.hparams.sampling_rate)

        return x
        
    def get_padding_length(self,input_length):

        input_length = math.ceil(input_length * self.hparams.resample)

  
        for layer in range(self.hparams.encoder_decoder["depth"]):                                        # encoder operation
            input_length = math.ceil((input_length - self.hparams.encoder_decoder["kernel_size"])/self.hparams.encoder_decoder["stride"])+1
            input_length = max(1,input_length)
        for layer in range(self.hparams.encoder_decoder["depth"]):                                        # decoder operaration
            input_length = (input_length-1) * self.hparams.encoder_decoder["stride"] + self.hparams.encoder_decoder["kernel_size"]
        input_length = math.ceil(input_length/self.hparams.resample)

        return int(input_length)

        

        








        