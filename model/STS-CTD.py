
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


from detection2.models.inception import InceptionBlock
from detection2.deeplearning.embedding import DataEmbedding, PositionalEncoding
from detection2.deeplearning.Attention import EncodeBlock


def bands_dropout(input, p):

    _, rows, cols = input.shape
    num_zeros = int(p * rows)
    r = torch.randperm(rows)[:num_zeros]
    input[:,r,:] = 0

    return input/(1 - p)

class Muti_kernel_conv1d(nn.Module):

    def __init__(self, d_model, C1_h):
        super(Muti_kernel_conv1d, self).__init__()

        self.conv1da = nn.Conv1d(d_model, C1_h, kernel_size= 1)
        self.conv1db = nn.Conv1d(d_model, C1_h, kernel_size= 3, padding=1)
        self.conv1dc = nn.Conv1d(d_model, C1_h, kernel_size= 5, padding=2)
    #    self.conv1dd = nn.Conv1d(d_model, C1_h, kernel_size= 7, padding=3)
        self.downsample = nn.Conv1d(d_model, C1_h, kernel_size= 1) 
        self.norm = nn.LayerNorm([C1_h, 27])
        self.GELU = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.dropout_ratio = 0
        #self.dropout_ratio = 0
        
    def forward(self, X):

        out = self.conv1da(X) + self.conv1db(X)  + self.conv1dc(X) # + self.conv1dd(X)
        #out = bands_dropout(self.conv1da(X),self.dropout_ratio) + bands_dropout(self.conv1db(X),self.dropout_ratio) + bands_dropout(self.conv1dc(X),self.dropout_ratio)
        #res = self.downsample(X)
        out = self.norm(bands_dropout(out, self.dropout_ratio))
        #out = self.GELU(out + res)
        #out = self.norm(self.dropout(out))
        #out = self.norm(out)
        #out = bands_dropout(self.norm(out), self.dropout_ratio)
        return out
    
class Transformer_Muti_kernel_Conv1d(nn.Module):

    def __init__(self, d_model, d_k, heads, dropout, norm_shape, num_encode,  ff_h, C1_h, C1_h2, C1_h3, seq_len):
        super(Transformer_Muti_kernel_Conv1d, self).__init__()    

        self.seq_len = seq_len
        self.pos_embedding = PositionalEncoding(d_model= d_model)
        self.encode_layer = EncodeBlock(d_model= d_model, d_k = d_k, heads = heads, dropout = dropout, norm_shape = norm_shape, ff_h = ff_h)
        self.layers = nn.Sequential()
        for _ in range(num_encode):
            self.layers.append(
                EncodeBlock(d_model= d_model,
                            d_k = d_k,
                            heads = heads,
                            dropout = dropout,
                            norm_shape = norm_shape,
                            ff_h = ff_h)
            )
            
        self.Mk_conv1d = nn.Sequential(
####################### L8 ###################################
            Muti_kernel_conv1d(d_model, C1_h),
            nn.GELU(),
            
            Muti_kernel_conv1d(C1_h, 36),
            nn.GELU(),
            
            Muti_kernel_conv1d(36, 30),
            nn.GELU(),     
                               
            Muti_kernel_conv1d(30, 24),
            nn.GELU(),
            
            Muti_kernel_conv1d(24, 18),
            nn.GELU(),
            
            Muti_kernel_conv1d(18, C1_h2),
            nn.GELU(),

            Muti_kernel_conv1d(C1_h2, C1_h3),
            nn.GELU(),

            Muti_kernel_conv1d(C1_h3, 1),
            nn.GELU(),
            )
        
    def forward(self, X):
        
        #x : batch, d_model, seq_len
        X = X.permute(0, 2, 1)

        #X : batch, seq_len, d_model
        X = X + self.pos_embedding(X)
        X = self.layers(X)

        #X --->:(batch, d_model, seq_len)
        X = X.permute(0,2,1)

        # multi_kernel 
        X = self.Mk_conv1d(X)
        #X = self.k(X)
        out  = X.permute(0, 2, 1)

        return out