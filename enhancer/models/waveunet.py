import torch.nn as nn


class WavenetDecoder(nn.Module):

    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int=5,
        padding:int=2,
        stride:int=1,
        dilation:int=1,
    ):
        super(WavenetDecoder,self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation),
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
        kernel_size:int=15,
        padding:int=7,
        stride:int=1,
        dilation:int=1,
    ):
        super(WavenetEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation),
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
        self,
        inp_channels:int=1,
        num_layers:int=12,
        initial_output_channels:int=24
    ):
        super(WaveUnet,self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        out_channels = initial_output_channels
        for layer in range(num_layers):

            encoder = WavenetEncoder(inp_channels,out_channels)
            self.encoders.append(encoder)

            inp_channels = out_channels
            out_channels += initial_output_channels
            if layer == num_layers -1 :
                decoder = WavenetDecoder(num_layers * initial_output_channels + inp_channels,inp_channels)
            else:
                decoder = WavenetDecoder(inp_channels+out_channels,inp_channels)

            self.decoders.insert(0,decoder)

        bottleneck_dim = num_layers * initial_output_channels
        self.bottleneck = nn.Sequential(
            nn.Conv1d(bottleneck_dim,bottleneck_dim, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        
        

    def forward(
        self,waveform
    ):
        
        for encoder in self.encoders:
            out = encoder(waveform)

        out = self.bottleneck(out)

        for decoder in self.decoders:
            out = decoder(out)

        return decoder