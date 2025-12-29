import math
import torch
import torch.nn as nn
import torch.optim as optim

class FFDModel(nn.Module):
    def __init__(self, args=None,mask_type='point',num_channel=1, patch_len=50, d_model=1024, num_heads=8, num_layers=8, dropout=0.1, max_seq_len=1000):
        """
        Args:
            input_size: feature dimension per sample point (e.g., 1)
            patch_size: number of sample points per token (e.g., 50)
            d_model: hidden dimension inside Transformer
            num_heads: number of heads in multi-head self-attention
            num_layers: number of Transformer Encoder layers
            dropout: dropout probability
            max_seq_len: maximum number of tokens (patch count) for positional embedding
        """
        super(FFDModel, self).__init__()
        self.args = args
        self.patch_len = patch_len
        self.d_model = d_model
        self.num_channel = num_channel
        self.max_seq_len = max_seq_len
        self.mask_type = mask_type

        # Map each patch (patch_len * input_channels) to d_model
        self.patch_embedding = nn.Linear(patch_len * self.num_channel, d_model)

        # Learnable mask token used to replace masked patches
        self.mask = nn.Parameter(torch.randn(1, 1, self.num_channel))

        # Learnable positional embeddings of shape (1, max_seq_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.xavier_uniform_(self.pos_embedding)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers: map Transformer features to forecasting targets
        self.feature_layer = nn.Linear(d_model, patch_len * 32)
        self.forecast_layer = nn.Linear(32, 1)
        # self.fc_out = nn.Linear(d_model, patch_len * self.num_channel)
        
    def get_mask(self, x):
        """
        Randomly select a continuous segment as the mask region. Mask length ratio is sampled
        from a normal distribution (mean 0.5, std 0.1) and clamped to [0.25, 0.75].
        There is a 50% probability to mask the tail, and 50% to mask a middle segment (25%~75%).
        
        Args:
            x (torch.Tensor): input time-series data, shape (L,) or (L, features)
            
        Returns:
            mask (torch.BoolTensor): boolean vector, shape (L,), True for masked positions
            masked_values (torch.Tensor): ground-truth values corresponding to masked positions
        """
        # Get sequence length; assume mask applies along the first dimension

        _, seq_len, _ = x.size()
        
        mask_length = 1
        # Sample mask length ratio and clamp to a valid range
        if self.mask_type == 'rate':
            mask_ratio = torch.normal(mean=0.2, std=0.03, size=(1,)).item()
            mask_ratio = max(0.1, min(mask_ratio, 0.3))
            mask_length = max(1, int(round(mask_ratio * seq_len)))
        elif self.mask_type == 'point':
            # Determine sampling range for mask length
            lower = self.patch_len
            upper = 2 * self.patch_len
            # If upper exceeds sequence length, cap at sequence length
            if upper > seq_len:
                upper = seq_len
            mask_length = torch.randint(lower, upper + 1, (1,)).item()

        mask = torch.zeros(seq_len, dtype=torch.bool)
        
        # Randomly select start index for mask segment
        if torch.rand(1).item() < 0.5:
            start_idx = seq_len - mask_length
        else:
            # Mask a continuous block in the middle of the sequence
            lower_bound = int(math.floor(0.25 * seq_len))
            upper_bound = int(math.floor(0.75 * seq_len)) - mask_length
            if upper_bound < lower_bound:
                start_idx = lower_bound
            else:
                start_idx = torch.randint(lower_bound, upper_bound + 1, (1,)).item()
        
        mask[start_idx: start_idx + mask_length] = True

        return mask

    def pretrain(self,x,mask_indices=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)   
        if mask_indices is None:
            # mask_token shape: (batch_size, num_patches, d_model)
            mask_indices = self.get_mask(x)
        # print(mask_indices.shape)
        masked_values = x[:,mask_indices,:].clone()
        x[:,mask_indices,:] = self.mask

        out = self.forward(x)

        # print(out.shape, masked_values.shape)
        return out[:,mask_indices,:],masked_values

    def finetune(self,x):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  
        out = self.backbone(x)
        return out

    def forecast(self,x,all_step):
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        predict_len = all_step*self.patch_len
        device = x.device
        x = torch.concat([x,torch.zeros(x.shape[0],predict_len,x.shape[2]).to(device)],dim=1)
        x[:,-predict_len:,:] = self.mask

        out = self.forward(x)
        return out[:,-predict_len:,:].squeeze(-1)

    def ffd(self,x):
        batch_size, T = x.shape
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)  
        out = self.backbone(x)

        return out
    
    def backbone(self,x):

        batch_size, seq_len, input_size = x.size()

        # Compute number of patches; truncate tail if not divisible by patch_len
        num_patches = seq_len // self.patch_len
        x = x[:, :num_patches * self.patch_len, :]
        # Reshape to (batch_size, num_patches, patch_len * input_channels)
        x = x.view(batch_size, num_patches, self.patch_len * input_size)

        # Project each patch to d_model
        x = self.patch_embedding(x)  # (batch_size, num_patches, d_model)
        x = x + self.pos_embedding[:, :num_patches, :]

        x = self.transformer_encoder(x)

        return x

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, input_channels)
        Returns:
            output tensor of shape (batch_size, seq_len, input_channels)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)   
        batch_size, seq_len, input_size = x.size()
        x = self.backbone(x)
        x = self.feature_layer(x).reshape(batch_size,seq_len,-1)
        x = self.forecast_layer(x)
        x = x.view(batch_size, -1,  input_size)
        return x
