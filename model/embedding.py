mport torch
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
    """Position Encoding"""
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(dropout)
        
        self.num_hiddens = d_model + (d_model % 2)*1 
        
        # create P
        self.P = torch.zeros((1, max_len, self.num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, self.num_hiddens, 2, dtype=torch.float32) / self.num_hiddens)
            
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.P = self.P[:, :, 0:(self.P.shape[2] -(d_model % 2)*1) ]

    def forward(self, X):
        X = self.P[:, :X.shape[1]].to(device)
        return  X


    
