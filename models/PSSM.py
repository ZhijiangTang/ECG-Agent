import math
import numpy as np
from torch import nn
import torch.nn.functional as F


class PSSM(nn.Module):
    def __init__(self,args=None, input_channels=1, d_model=32, **kwargs):
        super(PSSM, self).__init__()
        self.args = args
        self.input_channels = input_channels
        self.d_model = d_model

        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(self.downsample_block(input_channels, d_model))
        for i in range(3):
            self.encoder.append(self.downsample_block(d_model, d_model*2))
            d_model *= 2

        # Bottleneck
        self.bottleneck = self.conv_block(d_model, d_model*2)

        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(self.upconv_block(d_model*2, d_model))
            d_model = d_model // 2

        # Final layer
        self.final = nn.Conv1d(d_model*2, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def downsample_block(self, in_channels, out_channels):
        class DownsampleBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.patch = nn.AvgPool1d(kernel_size=2, stride=2)
                self.encoder = nn.Sequential(
                                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                                nn.BatchNorm1d(out_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                                nn.BatchNorm1d(out_channels),
                                nn.ReLU(inplace=True)        )    
                self.linear = nn.Linear(in_channels, out_channels)
                   
            
            def forward(self, x):                
                out = self.encoder(x)
                out =out+ self.linear(x.permute(0,2,1)).permute(0,2,1)
                out = self.patch(out)

                return out
        return DownsampleBlock(in_channels,out_channels)


    def upconv_block(self, in_channels, out_channels):
        class UpconvBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
                self.decoder = nn.Sequential(
                                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                                nn.BatchNorm1d(out_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
                                nn.BatchNorm1d(out_channels),
                                nn.ReLU(inplace=True))  
                self.linear = nn.Linear(out_channels, out_channels)
            
            def forward(self, x):
                x = self.upconv(x)
                
                out = self.decoder(x)
                out = out+self.linear(x.permute(0,2,1)).permute(0,2,1)
                return out

        return UpconvBlock(in_channels,out_channels)
    
    def padding(self,x):
        _, _,seq_len = x.shape

        target_len = 2**math.ceil(math.log2(seq_len))
        pad_len = target_len - seq_len
        if pad_len > 0:
            x = F.pad(x, (0, pad_len, 0, 0))
        
        return x,pad_len
    
    def finetune(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x,pad_len = self.padding(x)
        x = self.encoder[0](x)
        for i in range(1,len(self.encoder)):
            x=self.encoder[i](x)

        x = self.bottleneck(x)

        x = self.decoder[0](x)
        for i in range(1,len(self.decoder)):
            x = self.decoder[i](x)
        out = self.final(x)[:,:,:-pad_len]
        B,_,_ = out.shape
        out = out.reshape(B,-1)

        return out
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x,pad_len = self.padding(x)
        x_enc = [self.encoder[0](x)]
        for i in range(1,len(self.encoder)):
            x_enc.append(self.encoder[i](self.pool(x_enc[-1])))

        x_dec = self.bottleneck(self.pool(x_enc[-1]))

        x_dec = self.decoder[0](x_dec,x_enc[-1])
        for i in range(1,len(self.decoder)):
            x_dec = self.decoder[i](x_dec,x_enc[-i-1])
        out = self.final(x_dec).squeeze(1)

        return out