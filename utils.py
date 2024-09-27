
from typing import Any
import torch 
import os
import random
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import Dataset
from deeplearning.embedding import DataEmbedding
import rioxarray as rxr
import xarray as xr
from osgeo import gdal, osr
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import rasterio
from rasterio.transform import from_origin



def load_train_data(PATH, seq_len, dim_num):
    
    #dim_num = 7
    precipitation_data = rxr.open_rasterio(PATH).values
    print('precipitation_data: ',precipitation_data.shape)
    precipitation_data = torch.tensor(precipitation_data, dtype = torch.float32)
    shape = precipitation_data.shape
    data = precipitation_data.reshape(shape[0], -1).transpose(0,1)  # data.shape --> (weight * height, band_num)
    print(data.shape)
    # matrix shape： torch.Size([ num_pixels ])   torch.mean(data, 1) --> (weight * height, 1)  
    matrix = torch.mean(data, 1).flatten()

    #将去除label中的异常值
    _lable = data[:, -1]
    label_mask = (_lable >= 27) | (_lable < 1)

    #去除 nan 和 inf值
    matrix[torch.isinf(matrix) | torch.isnan(matrix)| label_mask] = 0
    non_negetive_indices = torch.nonzero(matrix)

    _data = data[non_negetive_indices.flatten(), :]  #_data: shape:[batch_num, seq_len * dim_num + 1] (b, 163)

    # x_data = _data[:,0: seq_len * dim_num].reshape(_data.shape[0], seq_len, -1).permute(0,2,1) # x_data : shape : [batch, dim, seq_len]
    # y_data = _data[:, -1].long() - 1 # y_data: shape : [batch, 1]
    # y_one_hot = one_hot_encode(y_data , 27) # --> [batch, seq_len]
    
    print('data_shape', _data.shape)

    return _data # shape: [batch_num, seq_len * dim_num + 1]

    

def one_hot_encode(labels, num_classes):
    """
    """
    # 使用torch.eye创建单位矩阵，并将对角线元素设为1，其他元素设为0
    eye_matrix = torch.eye(num_classes)
    eye_matrix[0] = 1/num_classes # 将第一行（索引为0）设为1，即对应标签为0的行设为全零
    eye_matrix[0][0] = 1
    # eye_matrix[0] = 1/(2*num_classes)
    # eye_matrix[0][0] = 0.5
    # 使用索引操作获取对应的独热编码
    one_hot_encoded = eye_matrix[labels]

    return one_hot_encoded

def load_train_data2(PATH, seq_len, dim_num):
    
    #dim_num = 6
    precipitation_data = rxr.open_rasterio(PATH).values

    precipitation_data = torch.tensor(precipitation_data, dtype = torch.float32)
    shape = precipitation_data.shape
    data = precipitation_data.reshape(shape[0], -1).transpose(0,1)  # data.shape --> (weight * height, band_num)
    print(data.shape)
    # matrix shape： torch.Size([ num_pixels ])   torch.mean(data, 1) --> (weight * height, 1)  
    matrix = torch.mean(data, 1).flatten()

    #将去除label中的异常值
    _lable = data[:, -1]
    label_mask = (_lable >= 27) | (_lable < 1)

    #去除 nan 和 inf值
    matrix[torch.isinf(matrix) | torch.isnan(matrix)| label_mask] = 0
    non_negetive_indices = torch.nonzero(matrix)

    _data = data[non_negetive_indices.flatten(), :]
    x_data = _data[:,0: seq_len * dim_num].reshape(_data.shape[0], seq_len, -1).permute(0,2,1)
    y_data = _data[:, -1].long() - 1
    y_one_hot = one_hot_encode(y_data , 27)
    
    print('y_label.shape:', y_one_hot.shape)
    print('x_data.shape :', x_data.shape)
    
    return x_data, y_one_hot, non_negetive_indices

###################################################################################
# ablation test for feature bands -----  0:VV 1:VH 2:RVI 3:DPSVI 4:CR 5:DPRVIs
def load_train_data3(PATH, seq_len, dim_num, is_remove_band, target_band):
    
    #dim_num = 6
    precipitation_data = rxr.open_rasterio(PATH).values
    
    
    #################### for two band #######################
    # if is_remove_band:
    #     if target_band == 0:
    #         label = precipitation_data[-1].reshape(1, precipitation_data.shape[1], precipitation_data.shape[2])
    #         print('label shape', label.shape)
    #         precipitation_data = np.delete(precipitation_data, np.s_[target_band::6], axis=0)
    #         precipitation_data = np.concatenate((precipitation_data, label), axis = 0)
    #     else:
    #         precipitation_data = np.delete(precipitation_data, np.s_[target_band::6], axis=0)
    ########################################################
    
    if is_remove_band:
        if target_band == 0:
            label = precipitation_data[-1].reshape(1, precipitation_data.shape[1], precipitation_data.shape[2])
            print('label shape', label.shape)
            precipitation_data = np.delete(precipitation_data, np.s_[target_band::6], axis=0)
            precipitation_data = np.concatenate((precipitation_data, label), axis = 0)
        elif target_band == 'VVVH':
            label = precipitation_data[-1].reshape(1, precipitation_data.shape[1], precipitation_data.shape[2])
            print('label shape', label.shape)
            precipitation_data = np.delete(precipitation_data, np.s_[0::6], axis=0)
            precipitation_data = np.delete(precipitation_data, np.s_[0::5], axis=0)
            precipitation_data = np.concatenate((precipitation_data, label), axis = 0)
        elif target_band == 'vvhh_except':
           # label = precipitation_data[-1].reshape(1, precipitation_data.shape[1], precipitation_data.shape[2])
            precipitation_data = np.delete(precipitation_data, np.s_[2::6], axis=0)
            precipitation_data = np.delete(precipitation_data, np.s_[2::5], axis=0)
            precipitation_data = np.delete(precipitation_data, np.s_[2::4], axis=0)
            precipitation_data = np.delete(precipitation_data, np.s_[2::3], axis=0)
            #precipitation_data = np.concatenate((precipitation_data, label), axis = 0)
        else:
            precipitation_data = np.delete(precipitation_data, np.s_[target_band::6], axis=0)        
        
        
        
    precipitation_data = torch.tensor(precipitation_data, dtype = torch.float32)
    shape = precipitation_data.shape
    data = precipitation_data.reshape(shape[0], -1).transpose(0,1)  # data.shape --> (weight * height, band_num)
    print(data.shape)
    # matrix shape： torch.Size([ num_pixels ])   torch.mean(data, 1) --> (weight * height, 1)  
    matrix = torch.mean(data, 1).flatten()

    #将去除label中的异常值
    _lable = data[:, -1]
    label_mask = (_lable >= 27) | (_lable < 1)

    #去除 nan 和 inf值
    matrix[torch.isinf(matrix) | torch.isnan(matrix)| label_mask] = 0
    non_negetive_indices = torch.nonzero(matrix)

    _data = data[non_negetive_indices.flatten(), :]
    x_data = _data[:,0: seq_len * dim_num].reshape(_data.shape[0], seq_len, -1).permute(0,2,1)
    y_data = _data[:, -1].long() - 1
    y_one_hot = one_hot_encode(y_data , 27)
    
    print('y_label.shape:', y_one_hot.shape)
    print('x_data.shape :', x_data.shape)
    
    return x_data, y_one_hot, non_negetive_indices
#######################################################################################

def Normalization(x_feature):
    
    mean = np.mean(x_feature.numpy(), axis=2).reshape(x_feature.shape[0], x_feature.shape[1], 1)
    std = np.std(x_feature.numpy(), axis=2).reshape(x_feature.shape[0], x_feature.shape[1], 1)
    x_norm = (x_feature - mean)/(std)
    
    return torch.tensor(x_norm)

class time_series_decode_paper(Dataset):
    
    def __init__(self, t , N , dx , dy):
        
        self.t = t
        self.N = N
        self.dx = dx
        self.dy = dy
        
        self.x = dx
        self.fx = dy
        
        print("x: ", self.dx.shape,
              'y:', self.dy.shape)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = (self.x[idx, :, :], self.fx[idx, :].unsqueeze(0))
        
        return sample
       
def train_epoch(net, train_dl, device, optimizer, criterion):
    
    net.train()
    train_loss = 0
    n = 0
    
    for step, (x,y) in enumerate(train_dl):
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        output = net(x)

        loss = criterion(output.squeeze(), y.cuda()[:, 0, :].float())
        loss.backward()
        optimizer.step()
        
        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]
        
    return train_loss/n

def save_img(output_filename, reference_filename, array):
    # 打开参考数据
    reference = gdal.Open(reference_filename)
    
    # 获取地理变换参数和坐标系
    geo_transform = reference.GetGeoTransform()
    projection = reference.GetProjection()

    # 获取矩阵尺寸
    rows, cols = array.shape

    # 创建新的 TIFF 数据集
    driver = gdal.GetDriverByName('GTiff')
    out_data = driver.Create(output_filename, cols, rows, 1, gdal.GDT_Float32)

    # 设置地理变换和坐标系
    out_data.SetGeoTransform(geo_transform)
    out_data.SetProjection(projection)
    

    # 写入数据
    out_data.GetRasterBand(1).WriteArray(array)

    # 保存并关闭文件
    out_data.FlushCache()
    out_data = None

def split_multiband_image_with_geo(input_image_path, output_path, block_size):
    dataset = gdal.Open(input_image_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    print(width,height,bands)
    geo_transform = dataset.GetGeoTransform()

    for i in range(0, width, block_size):
        for j in range(0, height, block_size):
            xoff = i
            yoff = j
            xsize = min(block_size, width - i)
            ysize = min(block_size, height - j)

            block_data = dataset.ReadAsArray(xoff, yoff, xsize, ysize)
            block_geo_transform = (geo_transform[0] + i * geo_transform[1],
                                   geo_transform[1], 0,
                                   geo_transform[3] + j * geo_transform[5],
                                   0, geo_transform[5])

            block_dataset = gdal.GetDriverByName('GTiff').Create(
                f"{output_path}/LZW_{j}_{i}.tif",
                xsize, ysize, bands, gdal.GDT_Float32
            )

            block_dataset.SetGeoTransform(block_geo_transform)
            block_dataset.SetProjection(dataset.GetProjection())

            for band in range(1, bands + 1):
                band_data = dataset.GetRasterBand(band).ReadAsArray(xoff, yoff, xsize, ysize)
                block_dataset.GetRasterBand(band).WriteArray(band_data)

            block_dataset.FlushCache()

    dataset = None


def get_result(input_path, row, col, net, param_path, device, is_remove_band, target_band):

    #open image
    z = rxr.open_rasterio(input_path).values
    #print('z.shape', z.shape)
    
    zz = z[:z.shape[0]-z.shape[0]%2,row[0]:row[1],col[0]:col[1]]    
    
    if is_remove_band:
        
        if target_band == 0:
            label = z[-1].reshape(1, z.shape[1], z.shape[2])
            z = np.delete(z, np.s_[target_band::6], axis=0)
            z = np.concatenate((z, label), axis = 0)
        elif target_band == 'vvvh':
            label = z[-1].reshape(1, z.shape[1], z.shape[2])
            z = np.delete(z, np.s_[0::6], axis=0)
            z = np.delete(z, np.s_[0::5], axis=0)
            z = np.concatenate((z, label), axis = 0)
        elif target_band == 'vvhh_except':
           # label = z[-1].reshape(1, z.shape[1], z.shape[2])
            z = np.delete(z, np.s_[2::6], axis=0)
            z = np.delete(z, np.s_[2::5], axis=0)
            z = np.delete(z, np.s_[2::4], axis=0)
            z = np.delete(z, np.s_[2::3], axis=0)
           # z = np.concatenate((z, label), axis = 0)
        else:
            z = np.delete(z, np.s_[target_band::6], axis=0)
        zz = z[:z.shape[0]-1,row[0]:row[1],col[0]:col[1]]
        
    print('z.shape',z.shape)

 
    #
    shape = zz.shape
    print(zz.shape)
    b = torch.tensor(zz).permute(1,2,0)
    c = b.reshape(-1,b.shape[2])
    x_feature = c.reshape(c.shape[0], 27 ,-1).permute(0,2,1)
    print('d.shape:',x_feature.shape)

    #normalization
    mean = np.mean(x_feature.numpy(), axis=2).reshape(x_feature.shape[0], x_feature.shape[1], 1)
    std = np.std(x_feature.numpy(), axis=2).reshape(x_feature.shape[0], x_feature.shape[1], 1)
    x_norm = (x_feature - mean)/(std)

    #load net
    net.load_state_dict(torch.load(param_path), strict= False)

    result = net((x_norm.float()).to(device))
    result = result.squeeze(1)
    result = torch.argmax(result, dim=1)
    result = result.reshape(shape[1],shape[2]).cpu().numpy()

    return result

def Accuarcy_percise(input_path, target_result, average, name, is_output, filename):

    test_data = rxr.open_rasterio(input_path).values
    print('precipitation_data: ',test_data.shape)
    #label = test_data[162,:326,:327] 
    label = test_data[162,:,:] 
    # 将矩阵展平为一维数组
    predicted_labels = target_result.flatten()
    predicted_labels = predicted_labels + 1
    true_labels = label.flatten()
    # 计算总体精度（Overall Accuracy）
    correctly_classified = np.sum(predicted_labels == true_labels)
    total_samples = len(true_labels)
    oa = correctly_classified / total_samples * 100
    precision = precision_score(true_labels, predicted_labels, average=average)  
    recall = recall_score(true_labels, predicted_labels, average=average)  
    ## 计算F1分数
    precision = precision_score(true_labels, predicted_labels, average=average) 
    f1 = f1_score(true_labels, predicted_labels, average=average)  
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print(f"Overall Accuracy (OA): {oa:.6f}%")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")
    print(conf_matrix)
    if is_output:
        conf_df = pd.DataFrame(conf_matrix)
        excel_file = f"C:/Users/minyu/Desktop/accuracy/paramE8_L8_{filename}_confusion_matrix.xlsx"
        conf_df.to_excel(excel_file, index=False)

        print(f"Confusion matrix saved to {excel_file}.")
    return conf_matrix

def count_tif_files(folder_path):
    # 初始化tif文件数量为0
    tif_count = 0
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否以.tif结尾
        if filename.endswith('.tif'):
            tif_count += 1
    return tif_count


def combine_train_data(output_path, path_first, *PATHs):

    _data = load_train_data(path_first, 27, 6)
    for path in PATHs:
        _data1 = load_train_data(path, 27, 6)
        _data = torch.cat([_data,  _data1], dim = 0)  
  
    sample_num = _data.shape[0]
    print('sample_num:',sample_num)
    
    random_indices = random.sample(range(sample_num), sample_num)

    #data shape: [batch_num, seq_len * dim + 1]
    dataset = _data[np.array([random_indices]).flatten(), :]
    print('dataset.shape: ',dataset.shape)

    row = int(np.sqrt(dataset.shape[0]))
    column = int((dataset.shape[0])/row)
    
    # (row, column, band_num)  --->  (band_num, row, column)
    dataset = torch.tensor(dataset, dtype = torch.float32)
    data = dataset[:row*column, :].reshape(row,column,163).permute(2,0,1)

    print('data_f: ',data.shape)
    print(50 * '#')
    
    # # 定义输出GeoTIFF文件的路径
    #output_path = 'E:/Sentinel-SAR/train_data/train_data_FJ.tif'
    transform = from_origin(0, 0, 1, 1)  # 定义仿射变换，这里以单位像素为单位
    band, height, width  = data.shape
    dtype = np.float32
    print(50 * '-')

    # 使用rasterio创建GeoTIFF文件并写入矩阵
    with rasterio.open(output_path, 'w', driver='GTiff', height = height, width = width,
                   count=band, dtype=dtype, transform=transform, crs=None) as dst:
        for i in range(band):
            dst.write(data[i], i + 1)
    
    print(f'test_data_f.tif save to: {output_path}')
    print(50 * '-')
    
    return None