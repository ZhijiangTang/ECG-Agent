
import math
import torch
from torch import nn
import torch.nn.functional as F

class BaseMLP(nn.Module):
    '''
    General MLP model with dropout
    '''
    def __init__(self, nodes_num=[500,2048,512], task_name='Forecast', dropout_prob=0.2):
        super().__init__()

        self.task_name = task_name
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.ReLU()
        
        self.layer = nn.ModuleList()
        for i in range(len(nodes_num)-1):
            self.layer.append(nn.Linear(nodes_num[i],nodes_num[i+1]))
        
        self.last_layer = None
        if self.task_name == 'Classification':
            self.last_layer = nn.Softmax(dim=-1)
        if self.task_name == 'BioClassification':
            self.last_layer = nn.Sigmoid()
    
    def forward(self, X):
        hid = self.layer[0](X)
        for i in range(len(self.layer)-1):
            hid = self.activation(hid)
            hid = self.dropout(hid)
            hid = self.layer[i+1](hid)
        
        if self.task_name in ['Classification','BioClassification']:
            hid = self.last_layer(hid)
        return hid


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        # First three Conv5x1 layers
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)

        # Attention layer
        self.attention1 = ChannelAttention(in_channels)

        # Residual branch
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_residual = nn.Conv1d(in_channels, in_channels, kernel_size=3,padding=1)

        # Second three Conv5x1 layers
        self.conv4 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv5 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv6 = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2)

        # Second Attention layer
        self.attention2 = ChannelAttention(in_channels)

    def forward(self, x):
        # First path: three Conv5x1 + Attention
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.attention1(out)

        # Second path: BN + ReLU + Conv2x1
        residual = self.bn(x)
        residual = self.relu(residual)
        residual = self.conv_residual(residual)

        # Residual connection
        out += residual

        # Second set of Conv5x1 + Attention
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.attention2(out)

        return out

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y

# Multi-Scale Branch
class MultiScaleBranch(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(MultiScaleBranch, self).__init__()
        self.residual_block = ResidualBlock(in_channels)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.residual_block(x)
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        return avg_out, max_out

# Multi-Scale Feature Network
class SECGNet(nn.Module):
    def __init__(self, args=None,in_channels=1):
        super(SECGNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 4, kernel_size=50, stride=1)
        self.conv2 = nn.Conv1d(4, 16, kernel_size=15, stride=1)
        self.conv3 = nn.Conv1d(16, 64, kernel_size=15, stride=1)

        self.branch1 = MultiScaleBranch(64, kernel_size=5)
        self.branch2 = MultiScaleBranch(64, kernel_size=3)

        self.fc = nn.Linear(256, 500)
        # self.softmax = nn.Softmax(dim=-1)

    def finetune(self, x):
        return self.forward(x)
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        avg1, max1 = self.branch1(x)
        avg2, max2 = self.branch2(x)

        # Concatenate features
        x = torch.cat([avg1, max1, avg2, max2], dim=1).squeeze(-1)
        # Fully connected layer
        x = self.fc(x)
        return x


class DENSECG(nn.Module):
    def __init__(self, args=None,input_channels=1):
        super(DENSECG, self).__init__()
        
        # CNN layers
        self.cnn1 = nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.cnn2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.cnn3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.AvgPool1d(kernel_size=5, stride=5)

        # BiLSTM layers
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.2)

        # Dense (fully connected) layer
        self.fc = nn.Linear(256, 20)
        # self.fc = nn.Linear(256, 20)
        # self.sigmoid = nn.Sigmoid()
    def finetune(self, x):
        return self.forward(x)
    def forward(self, x):
        # Input shape: (batch_size, input_channels, seq_len)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        # CNN layers
        x = self.pool1(F.relu(self.cnn1(x)))  # Shape: (batch_size, 32, seq_len)
        x = self.pool1(F.relu(self.cnn2(x)) ) # Shape: (batch_size, 64, seq_len)
        x = self.pool2(F.relu(self.cnn3(x)) ) # Shape: (batch_size, 128, seq_len)

        # print
        # Prepare for LSTM
        x = x.permute(0, 2, 1)    # Shape: (batch_size, seq_len // 2, 128)

        # BiLSTM layers
        x, _ = self.lstm1(x)      # Shape: (batch_size, seq_len // 2, 500)
        x, _ = self.lstm2(x)      # Shape: (batch_size, seq_len // 2, 250)

        # Dropout
        x = self.dropout(x)  # Take the last time step's output and apply dropout

        # Classification layer
        x = self.fc(x)
        B,_,_ = x.shape
        x = x.reshape(B,-1)
        # x = self.sigmoid(x)       # Shape: (batch_size, num_classes)
        
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,out_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert out_model % num_heads == 0, "out_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.out_model = out_model
        self.num_heads = num_heads
        self.d_k = out_model // num_heads
        
        self.W_q = nn.Linear(d_model, out_model)
        self.W_k = nn.Linear(d_model, out_model)
        self.W_v = nn.Linear(d_model, out_model)
        self.W_o = nn.Linear(out_model, out_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.out_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model,out_model, num_heads,dim_feedforward,  dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model,out_model=out_model, num_heads=num_heads)

        self.fc1 = nn.Linear(out_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, out_model)
        self.fc3 = nn.Linear(d_model, out_model)

        self.norm1 = nn.LayerNorm(out_model)
        self.norm2 = nn.LayerNorm(out_model)
        self.dropout = nn.Dropout(dropout)
    
    def feed_forward(self,x):
        return self.fc2(F.relu(self.fc1(x)))

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(self.fc3(x) + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x




