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
    def __init__(self, d_model, max_len=100):
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

class DataEmbedding(nn.Module):
    
    def __init__(self, c_in, d_model, dropout= 0.3):
        
        super(DataEmbedding, self).__init__()
        
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embeddding = PositionalEncoding(d_model= d_model)
        self.dropout = nn.Dropout(p = dropout)
        
    def forward(self, x):
        
        x = self.value_embedding(x) + self.position_embeddding(x)
        return self.dropout(x)
    
    
# x = torch.randn(10,8,7)

# k = DataEmbedding(7,6)
# xx =k(x)
# print(xx.shape)

#x = torch.randn(10, 6, 4)  # batch, seq_len, dim

# class PositionalEmbedding(nn.Module):

#     def __init__(self, d_model, max_len = 100):
#         super(PositionalEmbedding, self).__init__()
        
#         pe = torch.zeros(max_len, d_model).float()
#         pe.requires_grad =False
#         self.d_model = d_model + (d_model % 2)*1 
        
#         position = torch.arange(0, max_len).float().unsqueeze(1) # shape: [max_len, 1]
#         div_term = (torch.arange(0, self.d_model, 2).float() 
#                     * -(math.log(10000.0) / self.d_model)).exp()
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.P = self.P[:, :, 0:(self.P.shape[2] -(num_hiddens % 2)*1) ]
        
#         pe = pe.unsqueeze(0) # shape :[1, max_len, 1]
#         self.register_buffer('pe', pe)
        
#     def forward(self, x): 
#         #x.shape[1] mean max_seqence length
#         return self.pe[:,:x.shape[1]]