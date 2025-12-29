import math
import numpy as np
from torch import nn
import torch.nn.functional as F


class CustomizedModel(nn.Module):
    def __init__(self,args=None, input_channels=1, d_model=32, **kwargs):
        super(CustomizedModel, self).__init__()
        self.args = args
        self.input_channels = input_channels
        self.d_model = d_model

        self.encoder = nn.Sequential(nn.Linear(input_channels, d_model),nn.ReLU(inplace=True))
    
    def finetune(self, x):
        # Required method. Outputs extracted ECG features with shape [batch, d_model] or [batch, seq_len, d_model]
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        x = self.encoder(x)

        return x

    def forward(self, x):
        x = self.finetune(x)
        return x