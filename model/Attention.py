import sys
import os
curPath = os.path.abspath(os.path.dirname('detection2'))
sys.path.append(curPath)

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from detection2.deeplearning.embedding import TokenEmbedding, DataEmbedding


import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        
        d = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2))/np.sqrt(d)
        attn = nn.Softmax(dim =-1 )(scores)
        value = torch.matmul(self.dropout(attn), V)
        return value, attn
        


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.W_Q = nn.Linear(d_model, d_k * heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * heads, bias= False)
        self.W_V = nn.Linear(d_model, d_k * heads, bias=False)
        self.W_O = nn.Linear(d_k * heads, d_model, bias= False)
        self.ScaledDotProductAttention = ScaledDotProductAttention(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V):

        #Q : (Batch, seq_len, d_k * heads) -->(batch, heads, seq_len, d_k)
        Q = self.W_Q(input_Q)
        K = self.W_K(input_K)
        V = self.W_V(input_V)

        Q = rearrange(Q, 'b s (h d) -> b h s d', h = self.heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h = self.heads)
        V = rearrange(V, 'b s (h d) -> b h s d', h = self.heads)

        value, attn = self.ScaledDotProductAttention(Q, K, V)
        # value :(batch, heads, seq_len, d_k)--> (batch, seq_len, head*d_v)
        value = rearrange(value,' b h s d -> b s (h d)')
        
        output = self.W_O(value)

        return output

class Bands_Dropout(nn.Module):
    def __init__(self, p):
        super(Bands_Dropout, self).__init__()
        self.p = p

    def __call__(self, input):
        _, rows, cols = input.shape
        num_zeros = int(self.p * rows)
        r = torch.randperm(rows)[:num_zeros]
        input[:, r, :] = 0

        return input / (1 - self.p)
    
class FeedForward(nn.Module):

    def __init__(self, norm_shape,d_model, FF_h, dropout):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(norm_shape),
            nn.Linear(d_model, FF_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(FF_h, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, X):
        return self.net(X)

class AddNorm(nn.Module):

    def __init__(self, dropout, norm_shape):

        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.BandsDropout = Bands_Dropout(dropout)
        self.norm = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):

        return self.norm(X + self.dropout(Y))
    #    return self.norm(X + self.BandsDropout(Y))

class EncodeBlock(nn.Module):

    def __init__(self, d_model, d_k, heads, dropout, norm_shape, ff_h):

        super(EncodeBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, d_k, heads, dropout)
        self.addnorm1 = AddNorm(dropout, norm_shape)
        self.ffn = FeedForward(norm_shape, d_model, FF_h = ff_h, dropout= dropout)
        self.addnorm2 = AddNorm(dropout, norm_shape)

    def forward(self,X):
        
        Y = self.addnorm1(X.to(device), self.attention(X, X, X))

        return self.addnorm2(Y, self.ffn(Y))

class TransformerModel(nn.Module):

    def __init__(self, num_layers ,d_model, d_k, heads, dropout, norm_shape, ff_h):
        super(TransformerModel, self).__init__()
        self.encode_layers = nn.ModuleList(
            [EncodeBlock(d_model, d_k, heads, dropout, norm_shape, ff_h) for _ in range(num_layers)])
        
    def forward(self, src):
        layer_outputs = []
        for layer in self.encode_layers:
            src = layer(src)
            layer_outputs.append(src)
        # Concatenate layer outputs along the feature dimension
        concatenated_outputs = torch.cat(layer_outputs, dim=-1)
        return concatenated_outputs

# mutiheadAttention + addNorm + FFt + addNorm