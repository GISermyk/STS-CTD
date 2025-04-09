import sys
import os
curPath = os.path.abspath(os.path.dirname('detection2'))
sys.path.append(curPath)


import os, sys
import torch
import time
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from deeplearning.net import *
from load_data import *
from models.Conv1d_transformer import Transformer, transformer_conv1d,Transformer_Muti_kernel_Conv1d ,transformer_mlp, LSTM_conv1d
from models.Conv1d_transformer import *
from models.LSTM import BiLSTMModel, BiGRUModel
from models.TCN import TCN
from models.STSCDT import *
from deeplearning.params import *
from deeplearning.embedding import *


def main(device, save_condition, params_path, input_path):

    x_feature, y_one_hot, _ = load_train_data_revise(input_path, 27, 7)
    
    sample_num = x_feature.shape[0]

    # normalization      
    x_feature = Normalization3(x_feature)
    print('x_feature:',x_feature.shape)
    # print(x_feature[10, 6, :])
    # print(x_feature[10, 2, :])
    # print(y_one_hot[10])
    
    #----------------------------------------
    # x_feature = x_feature[:, :6, :]
    # x_feature = x_feature.permute(0, 2, 1)
    #----------------------------------------
    
    random_indices = random.sample(range(sample_num), sample_num)
    
    y_one_hot = y_one_hot/367.0

    dy = y_one_hot[np.array([random_indices]).flatten()]
    dx = x_feature[np.array([random_indices]).flatten(), :, :]
    #print('dx.shape',dx.shape)
    
    train_dataset = time_series_decode_paper(t = 27, N =sample_num, dx = dx, dy = dy)
    #shape : [batch, model_d, seq_len]

    train_dl = DataLoader(train_dataset,
                          batch_size = config.batch_size,
                          shuffle = True,
                          generator = torch.Generator(device='cpu'))
    
    ## **************************** load model ****************************************

    net = STS_CTD3(d_model = 128, d_k= 16, heads = 8, dropout=0.5, norm_shape = [27,128], num_encode = 6, ff_h= 256,
                   conv_channels=[128, 64, 32, 16, 1], seq_len = 27, bandDropout=0.2).to(device) 
    
    # net = STS_CTD3_MKD(d_model = 128, d_k= 16, heads = 8, dropout=0.5, norm_shape = [27,128], num_encode = 6, ff_h= 256,
    #               conv_channels=[128, 64, 32, 16, 1], seq_len = 27).to(device) 
    
    # net = STS_CTD3_SKBD(d_model = 128, d_k= 16, heads = 8, dropout=0.5, norm_shape = [27,128], num_encode = 6, ff_h= 256,
    #               conv_channels=[128, 64, 32, 16, 1], seq_len = 27, bandDropout = 0).to(device) 
    
    # net = STS_CTD3_MKNBP(d_model = 128, d_k= 16, heads = 8, dropout=0.5, norm_shape = [27,128], num_encode = 6, ff_h= 256,
    #                conv_channels=[128, 64, 32, 16, 1], seq_len = 27).to(device) 
    
    # net = BiLSTMModel_revise(input_size=6, hidden_size=256, num_layers=4, output_size=1).to(device)
    
    # net = BiGRUModel_revise(input_size=7, hidden_size=128, num_layers=4, output_size=1).to(device)

    # net = Transformer_revise(d_model = 128, d_k= 16, heads=8, dropout=0.5, norm_shape=[27, 128], ff_h=256, num_encoder=6, mlp = 256, mlp2=128).to(device)
    
    #net = Inception_time(in_channels=7, out_channel=32, kernel_sizes=[1, 3, 5], bottleneck_channels = 32).to(device)
    
    # net = TCN(input_size=7, output_size=1, num_channels=[72, 48, 36, 30, 24, 18, 12, 6, 6, 3]).to(device)
    
    #************************************************************************
    
    #optimizer = torch.optim.Adam(net.parameters(), lr = 0.005)
    #optimizer = torch.optim.Adam(net.parameters(), lr = 0.005)
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)
    #optimizer = torch.optim.Adam(net.parameters(), lr = 0.0005)
    #optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
    criterion = torch.nn.MSELoss()

    is_StepLR = True
    train_epoch_loss = []
    
    for e, epoch in enumerate(range(200)):
        
        #print('round ',e)
        train_loss = []
        l_train = train_epoch2(net, train_dl, device = device, optimizer = optimizer, criterion = criterion)
        train_loss.append(l_train)
        
        if is_StepLR:
            scheduler.step()
            
        epoch_mean_loss = np.mean(train_loss)
        
        if e % 10 == 0:
            
            with torch.no_grad():
                
                print(f"Epoch {e}: Train loss = {epoch_mean_loss:.6f}")
                
        train_epoch_loss.append(np.mean(epoch_mean_loss))
    
     # Save model
    if save_condition:
        
        if os.path.exists(params_path):
            None
        else:
            with open(params_path, 'w') as f:
                print('create param file : ',params_path)
        
        torch.save(net.state_dict(),params_path)
    
     

if __name__ == "__main__":
    
    params_path = 'E:/min/detection2/model_params/STS_CDT_Revise_all_3_result_e200_MKNBP.py'#STS_CDT_Revise_JSzz_e200_r.py'#paramsE8_b512.py'0
    #train_path = 'E:/Sentinel-SAR/Train_data_all/HB_SD_JS_SH_ZJ_FJ_GX_train_data7.tif'
    #train_path = 'E:/Sentinel-SAR/Train_data_all/other_train_data7_revise.tif'
    train_path = 'E:/Sentinel-SAR/Train_data_all/HB_SD_JS_SH_ZJ_FJ_GX_train_data7_revise.tif'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        start = time.time()

        main(device, True, params_path, train_path)
        end = time.time()
        print('runing time:', end-start)
        
    except(RuntimeError):
        print >> sys.stderr
        sys.exit(1)