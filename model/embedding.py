import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TokenEmbedding(nn.Module):
    
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels = c_in, out_channels= d_model,
                                   kernel_size=1,  padding_mode='circular', bias = False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu'
                )
        
    def forward(self, x):
        
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2).to(device)
        return x
    
        
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(dropout)
        
        self.num_hiddens = d_model + (d_model % 2)*1 
        
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, self.num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, self.num_hiddens, 2, dtype=torch.float32) / self.num_hiddens)
            
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.P = self.P[:, :, 0:(self.P.shape[2] -(d_model % 2)*1) ]

    def forward(self, X):
        X = self.P[:, :X.shape[1]].to(device)
        return  X

class DOY_PositionalEncoding(nn.Module):
    """DOY 位置编码"""
    def __init__(self, d_model):
        super(DOY_PositionalEncoding, self).__init__()
        
        self.num_hiddens = d_model
        
        Scaled_doy_values = torch.arange(1, 367, dtype=torch.float32, device=device).reshape(-1, 1) / 367.0
        X_pe = Scaled_doy_values / torch.pow(10000, torch.arange(0, self.num_hiddens, 2, dtype=torch.float32, device=device) / self.num_hiddens)
        
        self.P = torch.zeros((1, 367, self.num_hiddens), device=device)
        self.P[:, :, 0::2] = torch.sin(X_pe)
        self.P[:, :, 1::2] = torch.cos(X_pe)

    def forward(self, X):
        #print("---------------XSHAPE:------------",X.shape) #torch.Size([512, 27])
        # X.shape: (Batch, Seq_len, 1) -> squeeze(-1) -> (Batch, Seq_len)
        X = X.squeeze(-1).long()  # 确保 X 是整数索引
        #print("---------------XSHAPE:------------",X.shape)
        PE = self.P[:, X].squeeze(0).to(device)  # 确保 PE 在同一设备上  #PE: Batch,, Seq, model
        #print('----------PE.shape-----------:', PE.shape)
        return PE

    
