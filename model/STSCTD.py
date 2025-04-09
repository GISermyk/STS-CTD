import sys
import os
curPath = os.path.abspath(os.path.dirname('detection2'))
sys.path.append(curPath)

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
from models.inception import InceptionBlock

from deeplearning.embedding import PositionalEncoding  ,DOY_PositionalEncoding
from deeplearning.Attention import EncodeBlock
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def bands_dropout(input, p):

#     _, rows, cols = input.shape
#     num_zeros = int(p * rows)
#     r = torch.randperm(rows)[:num_zeros]
#     input[:,r,:] = 0

#     return input/(1 - p)

def bands_dropout(input, p):
    
    _, rows, cols = input.shape
    num_zeros = int(p * rows)
    r = torch.randperm(rows)[:num_zeros]
    output = input.clone()
    output[:, r, :] = 0
    return output / (1 - p)

class Muti_kernel_conv1d(nn.Module):

    def __init__(self, d_model, C1_h, bandDropout):
        super(Muti_kernel_conv1d, self).__init__()

        self.conv1da = nn.Conv1d(d_model, C1_h, kernel_size= 1)
        self.conv1db = nn.Conv1d(d_model, C1_h, kernel_size= 3, padding=1)
        self.conv1dc = nn.Conv1d(d_model, C1_h, kernel_size= 5, padding=2)
        
        self.norm = nn.LayerNorm([C1_h, 27])
        self.GELU = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.dropout_ratio = bandDropout

        
    def forward(self, X):

        out = self.conv1da(X) + self.conv1db(X)  + self.conv1dc(X) # + self.conv1dd(X)

        out = self.norm(bands_dropout(out, self.dropout_ratio))
  
        return out

#-----------------------------learnable PE-----------------------------------------
class STS_CTD3(nn.Module):

    def __init__(self, d_model, d_k, heads, dropout, norm_shape, num_encode, ff_h, conv_channels, seq_len, bandDropout):
        super(STS_CTD3, self).__init__()    

        self.seq_len = seq_len
        self.embedding = nn.Linear(7, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 27, d_model))  # Learnable PE
        # self.pos_encoding = PositionalEncoding(d_model= d_model)
        self.encode_layer = EncodeBlock(d_model= d_model, d_k = d_k, heads = heads, dropout = dropout, norm_shape = norm_shape, ff_h = ff_h)
        #self.fc = torch.nn.Linear(seq_len, 1)
        self.fc = nn.Linear(27, 1)      
        self.layers = nn.ModuleList([
            EncodeBlock(d_model= d_model,
                        d_k = d_k,
                        heads = heads,
                        dropout = dropout,
                        norm_shape = norm_shape,
                        ff_h = ff_h)
            for _ in range(num_encode)
        ])    
        
        self.convlayers = []
        for in_ch, out_ch in zip(conv_channels[:-1], conv_channels[1:]):  # Pair input and output channels
            self.convlayers.append(Muti_kernel_conv1d(in_ch, out_ch, bandDropout))
            self.convlayers.append(nn.GELU())
        self.Mk_conv1d =  nn.Sequential(*self.convlayers)

    def forward(self, X):
        
        # X.Shape : batch, d_model, seq_len (B, 7, d_model) -->Include: Feature Bands (B , 6, d_model) and DOY (B , 1, d_model)
        
        #X : batch, d_model, seq_len  ---># X : batch, seq_len, d_model
        X = X.permute(0, 2, 1)
  
        X = self.embedding(X)
        #print('------X.shape', X.shape)
        X = X + self.pos_embedding

        for layer in self.layers:
            X = layer(X)
 
        #X --->:(batch, d_model, seq_len)
        X = X.permute(0,2,1)

        # multi_kernel 
        X = self.Mk_conv1d(X)
        #X = self.k(X)
        out  = X.permute(0, 2, 1).squeeze(-1)
        #print('--------out shape:', out.shape)
        out = self.fc(out)
        #print('--------out shape:', out.shape)
        return out

#---------------------------------- Fixed PE ------------------------------------------
class STS_CTD_withFixedPE(nn.Module):
    """
        Args:
            d_model: Input feature dimension.
            d_k: Dimension of keys/queries in attention mechanism.
            heads: Number of attention heads.
            dropout: Dropout rate.
            norm_shape: Shape for layer normalization.
            num_encode: Number of encoder layers.
            ff_h: Hidden size of feed-forward layers.
            conv_channels: List of integers defining the channel sizes for the multi-kernel convolutional layers.
            seq_len: Sequence length.
            
    """
    def __init__(self, d_model, d_k, heads, dropout, norm_shape, num_encode, ff_h, conv_channels, seq_len, bandDropout):
        super(STS_CTD_withFixedPE, self).__init__()    

        self.seq_len = seq_len
        self.embedding = nn.Linear(6, d_model)
        self.pos_encoding = DOY_PositionalEncoding(d_model= d_model)
        self.encode_layer = EncodeBlock(d_model= d_model, d_k = d_k, heads = heads, dropout = dropout, norm_shape = norm_shape, ff_h = ff_h)
        self.fc = nn.Linear(27, 1)
        self.layers = nn.ModuleList([
            EncodeBlock(d_model= d_model,
                        d_k = d_k,
                        heads = heads,
                        dropout = dropout,
                        norm_shape = norm_shape,
                        ff_h = ff_h)
            for _ in range(num_encode)
        ])    
        
        self.convlayers = []
        for in_ch, out_ch in zip(conv_channels[:-1], conv_channels[1:]):  # Pair input and output channels
            self.convlayers.append(Muti_kernel_conv1d(in_ch, out_ch, bandDropout))
            self.convlayers.append(nn.GELU())
        self.Mk_conv1d =  nn.Sequential(*self.convlayers)

    def forward(self, X):
        
        # X.Shape : batch, d_model, seq_len (B, 7, d_model) -->Include: Feature Bands (B , 6, d_model) and DOY (B , 1, d_model)
        
        #X : batch, d_model, seq_len  ---># X : batch, seq_len, d_model
        X = X.permute(0, 2, 1)
        DOY, FBs = X[:, : ,6], X[:, :, :6]
        
        #X : batch, seq_len, d_model
        X = self.embedding(FBs)
        #print('------X.shape', X.shape)
        X = X + self.pos_encoding(DOY)

        for layer in self.layers:
            X = layer(X)
 
        #X --->:(batch, d_model, seq_len)
        X = X.permute(0,2,1)

        # multi_kernel 
        X = self.Mk_conv1d(X)
        #X = self.k(X)
        out  = X.permute(0, 2, 1).squeeze(-1)
        #print('--------out shape:', out.shape)
        out = self.fc(out)
        #print('--------out shape:', out.shape)
        return out

    
