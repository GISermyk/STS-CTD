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
from detection2.load_data import *
from models.Conv1d_transformer import Conv1d_transformerEncode, transformer_conv1d,Transformer_Muti_kernel_Conv1d ,transformer_mlp, LSTM_conv1d
from models.Conv1d_transformer import *
from models.LSTM import BiLSTMModel, BiGRUModel
from models.TCN import TCN
from deeplearning.params import *
from deeplearning.embedding import *


def main(device, save_condition, params_path, input_path):

    x_feature, y_one_hot,_ = load_train_data2(input_path, 27, 6)
    
    ####  ablation test for feature bands -----  0:VV 1:VH 2:RVI 3:DPSVI 4:CR 5:DPRVIs  ######
    # x_feature, y_one_hot,_ = load_train_data3(input_path, 27, 2, is_remove_band=True, target_band = 'vvhh_except')
    
    ###########################################################
    
    # x_feature, y_one_hot ,_ = load_train_data2(path_1st, 27, 6)
    # for path in paths:
    #     x_feature1, y_one_hot1 ,_ = load_train_data2(path, 27, 6)
    #     x_feature = torch.cat([x_feature, x_feature1], dim = 0)
    #     y_one_hot = torch.cat([y_one_hot, y_one_hot1], dim = 0)
    
    sample_num = x_feature.shape[0]

    # normalization      
    x_feature = Normalization(x_feature)
    print('x_feature:',x_feature.shape)
    
    random_indices = random.sample(range(sample_num), config.train_num)
    dy = y_one_hot[np.array([random_indices]).flatten(), :]
    dx = x_feature[np.array([random_indices]).flatten(), :, :]
    #print('dx.shape',dx.shape)
    
    train_dataset = time_series_decode_paper(t = 27, N =config.train_num, dx = dx, dy = dy)
    #shape : [batch, model_d, seq_len]

    train_dl = DataLoader(train_dataset,
                          batch_size = config.batch_size,
                          shuffle = True,
                          generator = torch.Generator(device='cpu'))
    
    ## **************************** load model ****************************************

    # net = transformer_mlp(d_model = 7, d_k=7, heads=4,dropout=0.5, norm_shape=[27,7], ff_h=14,
    #                           num_encode= 4, mlp_h=21, mlp_h2=7).to(device)                           

    net = Transformer_Muti_kernel_Conv1d(d_model = 6, d_k= 6, heads = 8, dropout=0.5, norm_shape = [27,6], num_encode = 8,
                                         ff_h= 12, C1_h = 48, C1_h2 = 12, C1_h3= 6, seq_len = 27).to(device)
    # net = Transformer_Muti_kernel_Conv1d(d_model = 2, d_k= 2, heads = 8, dropout = 0.5, norm_shape = [27,2], num_encode = 8,
    #                                      ff_h= 12, C1_h = 48, C1_h2 = 12, C1_h3= 6, seq_len = 27).to(device)
    # net = Transformer_Muti_kernel_Conv1d(d_model = 6, d_k= 6, heads = 8, dropout=0.5,shape = [27,6], num_encode = 8,
    #                                      ff_h= 12, C1_h = 48, C1_h2 = 12,  norm_C1_h3= 6, seq_len = 27).to(device)
    
    # net = Transformer_Muti_kernel_Conv1d(d_model = 6, d_k=6, heads = 12, dropout=0.5, norm_shape = [27,6], num_encode = 8,
    #                                      ff_h= 12, C1_h = 24, C1_h2 = 12, C1_h3= 6, seq_len = 27).to(device) 
    # net = Transformer_Muti_kernel_Conv1d(d_model = 6, d_k=6, heads = 8, dropout=0.5, norm_shape = [27,6], num_encode = 8,
    #                                     ff_h= 12, C1_h = 48, C1_h2 = 12, C1_h3 = 6, seq_len = 27).to(device)
    
    # net = Transformer_Muti_kernel_Conv1d_DP(d_model = 6, d_k=6, heads = 8, dropout=0.5, norm_shape = [27,6], num_encode = 8,
    #                                         ff_h= 12, C1_h = 48, C1_h2 = 12, C1_h3 = 6, seq_len = 27).to(device)
    # net = transformer_mlp(d_model=6, d_k=6, heads=8, dropout=0.5, norm_shape=[27,6], num_encode=8,
    #                       ff_h=12, mlp_h= 48, mlp_h2= 12, mlp_h3=6).to(device)
    # #net = TCN(input_size=6, output_size=1, num_channels=[72, 36, 30, 24, 18, 12, 6, 6, 3]).to(device) #T2
    # #net = TCN(input_size=6, output_size=1, num_channels=[72, 48, 36, 30, 24, 18, 12, 6, 6, 3]).to(device) 
    # net = BiLSTMModel(input_size=6, hidden_size=256, num_layers=4, output_size=1).to(device)
    #net = BiGRUModel(input_size=6, hidden_size=12, num_layers=4, output_size=1).to(device)
    # net = BiGRUModel(input_size=6, hidden_size=128, num_layers=4, output_size=1).to(device)
    # net = Inception_time(in_channels=6, out_channel=32, kernel_sizes=[1, 3, 5], bottleneck_channels = 32).to(device)
    
    #************************************************************************
    
    
    #optimizer = torch.optim.Adam(net.parameters(), lr = 0.005)
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    #optimizer = torch.optim.Adam(net.parameters(), lr = 0.0005)
    #optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
    criterion = torch.nn.MSELoss()


    train_epoch_loss = []
    
    for e, epoch in enumerate(range(800)):
        
        train_loss = []
        l_train = train_epoch(net, train_dl, device=device, optimizer = optimizer, 
                              criterion = criterion)
        train_loss.append(l_train)
        
        if e % 40 == 0:
            with torch.no_grad():
                
                print("Epoch {}: Train loss={} \t ".format(e,np.mean(train_loss)))
                
                train_epoch_loss.append(np.mean(train_loss))
    
     # Save model
    if save_condition:
        
        if os.path.exists(params_path):
            None
        else:
            with open(params_path, 'w') as f:
                print('create param file : ',params_path)
        
        torch.save(net.state_dict(),params_path)
    
     

if __name__ == "__main__":
    
    params_path = 'E:/min/detection2/model_params/paramsE8_b512.py'
    train_path = 'E:/Sentinel-SAR/Train_data_all/HB_SD_JS_SH_ZJ_FJ_GX_train_data7.tif'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        start = time.time()

        main(device, True, params_path, train_path)
        end = time.time()
        print('runing time:', end-start)
        
    except(RuntimeError):
        print >> sys.stderr
        sys.exit(1)