import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import sys
import numpy as np
import scipy.io as io
import data_reader
from GNN import split_data_graph as split_data
from GNN import create_graph
from GNN import dr_slic
import warnings
warnings.filterwarnings("ignore")

def load_data():
    data = data_reader.IndianRaw().normal_cube
    data_gt = data_reader.IndianRaw().truth
    data_gt = data_gt.astype('int')
    return data,data_gt
data, data_gt = load_data()
class_num = np.max(data_gt)
gt_reshape = np.reshape(data_gt, [-1])
samples_type = ['ratio','same_num'][0]

train_ratio = 0.001
val_ratio = 0.001
train_num = 10
val_num = class_num
superpixel_scale = 100
# dataset_name = "indian_"
path_data = None
height,width,bands = data.shape
# print(height,width,bands)

#split data
train_index,val_index,test_index = split_data.split_data(gt_reshape,class_num,train_ratio,val_ratio,train_num,val_num,samples_type)#划分训练集、验证集和测试集,得到属于三种集的坐标
train_samples_gt,test_samples_gt,val_samples_gt = create_graph.get_label(gt_reshape,train_index,val_index,test_index)#ground_truth划分为训练集、验证机和测试集，维度是21025，根据坐标得到属于各种集的类别
train_gt = np.reshape(train_samples_gt,[height,width])
test_gt = np.reshape(test_samples_gt,[height,width])
val_gt = np.reshape(val_samples_gt,[height,width])#维度都是[145,145],reshape之后，对应的训练集，测试集，验证集都显现出来了。训练集上，测试集和验证集对应位置为0

train_samples_gt_onehot = create_graph.label_to_one_hot(train_gt,class_num)
test_samples_gt_onehot = create_graph.label_to_one_hot(test_gt,class_num)
val_samples_gt_onehot = create_graph.label_to_one_hot(val_gt,class_num)#维度都是[145,145,16]，假如(x,y)属于第9类，那么[x,y,9]即为1
train_samples_gt_onehot = np.reshape(train_samples_gt_onehot,[-1,class_num]).astype(int)
test_samples_gt_onehot = np.reshape(test_samples_gt_onehot,[-1,class_num]).astype(int)
val_samples_gt_onehot = np.reshape(val_samples_gt_onehot,[-1,class_num]).astype(int)#维度都是[21025,16]

train_label_mask,test_label_mask,val_label_mask = create_graph.get_label_mask(train_samples_gt,test_samples_gt,val_samples_gt,data_gt,class_num)#[21025,16]

ls = dr_slic.LDA_SLIC(data,np.reshape(train_samples_gt,[height,width]),class_num - 1)
Q,S,A,Seg = ls.simple_superpixel(scale=superpixel_scale)#Q为关联矩阵，21025====>196,每个像素对应着一个坐标为1

# # print(test_label_mask.shape)
# print(A.shape)
# # print(predict.shape)
#
# io.savemat('/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/Salinas/train_samples_gt.mat', {'train_samples_gt':train_samples_gt})
# io.savemat('/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/Salinas/test_samples_gt.mat', {'test_samples_gt':test_samples_gt})
# io.savemat('/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/Salinas/val_samples_gt.mat', {'val_samples_gt':val_samples_gt})
# print(1)
# io.savemat('/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/Salinas/train_samples_gt_onehot.mat', {'train_samples_gt_onehot':train_samples_gt_onehot})
# io.savemat('/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/Salinas/test_samples_gt_onehot.mat', {'test_samples_gt_onehot':test_samples_gt_onehot})
# io.savemat('/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/Salinas/val_samples_gt_onehot.mat', {'val_samples_gt_onehot':val_samples_gt_onehot})
# print(1)
# io.savemat('/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/Salinas/train_label_mask.mat', {'train_label_mask':train_label_mask})
# io.savemat('/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/Salinas/test_label_mask.mat', {'test_label_mask':test_label_mask})
# io.savemat('/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/Salinas/val_label_mask.mat', {'val_label_mask':val_label_mask})
# print(1)
# io.savemat('/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/Salinas/A.mat', {'A':A})
# io.savemat('/home/project/GNN_for_HSI/NL_GNN_for_HSI/Datasets/Salinas/Q.mat', {'Q':Q})
# print(1)

class Denoise(nn.Module):
    def __init__(self,channel:int,out_channels:int,layers:int):
        super(Denoise,self).__init__()
        self.channel = channel
        self.out_channels = out_channels
        self.layers = layers
        self.CNN_denoise = nn.Sequential()
        for i in range(self.layers):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel,self.out_channels,kernel_size=(1,1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.out_channels))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
    def forward(self,x):
        return self.CNN_denoise(x)
class GCNconv(nn.Module):
    def __init__(self,in_channels,out_channels,bias = True):
        super(GCNconv,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels,out_channels))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)

    def forward(self,x,adj):
        support = torch.matmul(x,self.weight)###
        output = torch.matmul(adj,support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_channels) + '->' + str(self.out_channels) + ')'
class GCN(nn.Module):
    def __init__(self,in_channels,in_hid,classes,Q,out,dropout=0.5):
        super(GCN,self).__init__()
        self.gc1 = GCNconv(in_channels=in_channels,out_channels=in_hid)
        self.gc2 = GCNconv(in_hid,classes)
        self.bn = nn.BatchNorm1d(num_features=in_channels)
        self.bn1 = nn.BatchNorm1d(num_features=in_hid)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q,[21025,196]
        self.Q = Q
        self.lin = nn.Linear(classes, out)
        self.Denoise = Denoise(channel=200,out_channels=200,layers=2)
    def forward(self,x,adj):
        (h,w,c) = x.shape
        # noise = self.Denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        # noise = torch.squeeze(noise, 0).permute([1, 2, 0])  # [145,145,128]
        re_x = x.reshape([h*w,-1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), re_x)
        x = superpixels_flatten
        x = self.bn(x)
        x = torch.relu(self.gc1(x,adj))
        x = self.bn1(x)
        x = self.gc2(x,adj)
        x = torch.matmul(self.Q,x)
        x = self.lin(x)
        return torch.softmax(x,dim=1)

# class GCN(nn.Module):
#     def __init__(self,in_channels,in_hid,classes,dropout=0.5):
#         super(GCN,self).__init__()
#         self.gc1 = GCNconv(in_channels=in_channels,out_channels=in_hid)
#         self.gc2 = GCNconv(in_hid,classes)
#         self.bn = nn.BatchNorm1d(num_features=in_channels)
#         self.bn1 = nn.BatchNorm1d(num_features=in_hid)
#         self.dropout = nn.Dropout(p=dropout)
#     def forward(self,x,adj):
#         x = self.bn(x)
#         x = F.leaky_relu(self.gc1(x,adj))
#         x = self.bn1(x)
#         x = self.gc2(x,adj)
#         return x
# class Res_GCN(nn.Module):
#     def __init__(self,in_channels,out_channels,nhid,Q):
#         super(Res_GCN, self).__init__()
#         self.gcn1 = GCN(in_channels=in_channels,in_hid=nhid,classes=in_channels)
#         self.gcn2 = GCN(in_channels=in_channels,in_hid=nhid,classes=out_channels)
#         self.bn1 = nn.BatchNorm1d(num_features=in_channels)
#         self.bn2 = nn.BatchNorm1d(num_features=in_channels)
#         self.bn3 = nn.BatchNorm1d(num_features=out_channels)
#         self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q,[21025,196]
#         self.Q = Q
#         self.lin = nn.Linear(out_channels, 16)
#         self.Denoise = Denoise(channel=103, out_channels=103, layers=2)
#     def reset_parameters(self):
#         self.gcn1.reset_parameters()
#         self.gcn2.reset_parameters()
#     def forward(self,x,A):
#         (h,w,b)=x.shape
#         # noise = self.Denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
#         # noise = torch.squeeze(noise, 0).permute([1, 2, 0])  # [145,145,128]
#         re_x = x.reshape([h * w, -1])
#         superpixels_flatten = torch.mm(self.norm_col_Q.t(), re_x)
#         x = superpixels_flatten
#         x = self.bn1(x)
#         y = F.leaky_relu(self.bn2(self.gcn1(x, A)))
#         x = torch.add(x, y)
#         x = F.leaky_relu(self.bn3(self.gcn2(x, A)))
#         x = torch.matmul(self.Q, x)
#         x = self.lin(x)
#         return torch.softmax(x,dim=1)