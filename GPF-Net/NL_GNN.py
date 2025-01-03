import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import sys
from AF2GNN import AF2GNN
# from torch_geometric.nn import GCNConv,GATConv
'''
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
    def __init__(self,in_channels,in_hid,classes,Q,dropout=0.5):
        super(GCN,self).__init__()
        self.gc1 = GCNconv(in_channels=in_channels,out_channels=in_hid)
        self.gc2 = GCNconv(in_hid,classes)
        self.bn = nn.BatchNorm1d(num_features=in_channels)
        self.bn1 = nn.BatchNorm1d(num_features=in_hid)
        self.Denoise = Denoise(channel=200,layers=2)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q,[21025,196]
        self.Q = Q
    def forward(self,x,adj):
        (h,w,c) = x.shape
        x = self.Denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        re_x = x.reshape([h*w,-1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), re_x)
        # print(superpixels_flatten.shape)
        x = superpixels_flatten
        # print(x.shape)
        x = self.bn(x)
        x = torch.relu(self.gc1(x,adj))
        x = self.bn1(x)
        x = self.gc2(x,adj)
        x = torch.matmul(self.Q,x)
        return torch.softmax(x,dim=1)

class NL_GCN(nn.Module):
    def __init__(self,in_channels,hidden,out_channels,class_num,Q):
        super(NL_GCN, self).__init__()
        self.conv1 = GCNconv(in_channels,hidden)
        self.conv2 = GCNconv(hidden,out_channels)
        self.proj = nn.Linear(out_channels,1)
        self.kernel = 5
        self.conv1d = nn.Conv1d(out_channels,out_channels,self.kernel,padding=int((self.kernel - 1)/2))
        self.conv1d_2 = nn.Conv1d(out_channels,out_channels,self.kernel,padding=int((self.kernel - 1)/2))
        self.lin = nn.Linear(2*out_channels,class_num)
        self.bn = nn.BatchNorm1d(num_features=in_channels,affine=False)
        self.bn1 = nn.BatchNorm1d(num_features=hidden,affine=False)
        # self.lin2 = nn.Linear(out_channels,class_num)
        self.training = True
        self.Q = Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q,[21025,196]
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.proj.reset_parameters()
        self.conv1d.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin.reset_parameters()
    def forward(self,x,adj):
        (h,w,b) = x.shape
        re_x = x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), re_x)
        # x = superpixels_flatten
        x = self.bn(superpixels_flatten)
        x = F.relu(self.conv1(x,adj))
        x = self.bn1(x)
        x1 = self.conv2(x,adj)
        g_score = self.proj(x1)#[nodes_num,1]
        g_score_sorted,sort_idx = torch.sort(g_score,dim=0)
        _,inverse_idx = torch.sort(sort_idx,dim=0)
        sorted_x = g_score_sorted*x1[sort_idx].squeeze()
        sorted_x = torch.transpose(sorted_x,0,1).unsqueeze(0) # [1, dataset.num_classes, num_nodes]
        sorted_x = F.relu(self.conv1d(sorted_x))
        sorted_x = F.dropout(sorted_x,p=0.5,training=self.training)
        sorted_x = self.conv1d_2(sorted_x)
        sorted_x = torch.transpose(sorted_x.squeeze(),0,1)# [num_nodes, dataset.num_classes]
        x2 = sorted_x[inverse_idx].squeeze()  # [num_nodes, dataset.num_classes]

        out = torch.cat([x1,x2],dim=1)
        out = F.dropout(out,p=0.5,training=self.training)
        out = self.lin(out)
        GCN_result = torch.matmul(self.Q,out)
        return F.softmax(GCN_result,dim=1)
class GCN1(nn.Module):
    def __init__(self,in_channels,in_hid,classes,dropout=0.5):
        super(GCN1,self).__init__()
        self.gc1 = GCNconv(in_channels=in_channels,out_channels=in_hid)
        self.gc2 = GCNconv(in_hid,classes)
        self.bn = nn.BatchNorm1d(num_features=in_channels)
        self.bn1 = nn.BatchNorm1d(num_features=in_hid)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,x,adj):
        x = self.bn(x)
        x = F.leaky_relu(self.gc1(x,adj))
        x = self.bn1(x)
        x = self.gc2(x,adj)
        return x
class Res_GCN(nn.Module):
    def __init__(self,in_channels,out_channels,nhid):
        super(Res_GCN, self).__init__()
        self.gcn1 = GCN1(in_channels=in_channels,in_hid=nhid,classes=in_channels)
        self.gcn2 = GCN1(in_channels=in_channels,in_hid=nhid,classes=out_channels)
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.bn2 = nn.BatchNorm1d(num_features=in_channels)
        self.bn3 = nn.BatchNorm1d(num_features=out_channels)
    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
    def forward(self,x,A):
        x = self.bn1(x)
        y = F.leaky_relu(self.bn2(self.gcn1(x, A)))
        x = torch.add(x, y)
        x = F.leaky_relu(self.bn3(self.gcn2(x, A)))
        return x

class NL_Res_GCN(nn.Module):
    def __init__(self,in_channels,hidden,out_channels,class_num,Q):
        super(NL_Res_GCN, self).__init__()
        self.conv1 = Res_GCN(in_channels,out_channels,hidden)
        self.proj = nn.Linear(out_channels,1)
        self.kernel = 5
        self.conv1d = nn.Conv1d(out_channels,out_channels,self.kernel,padding=int((self.kernel - 1)/2))
        self.conv1d_2 = nn.Conv1d(out_channels,out_channels,self.kernel,padding=int((self.kernel - 1)/2))
        self.lin = nn.Linear(2*out_channels,class_num)
        self.bn = nn.BatchNorm1d(num_features=in_channels,affine=False)
        self.bn1 = nn.BatchNorm1d(num_features=hidden,affine=False)
        # self.lin2 = nn.Linear(out_channels,class_num)
        self.training = True
        self.Q = Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q,[21025,196]
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.proj.reset_parameters()
        self.conv1d.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin.reset_parameters()
    def forward(self,x,adj):
        (h,w,b) = x.shape
        re_x = x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), re_x)
        # x = superpixels_flatten
        x = self.bn(superpixels_flatten)
        # x = F.relu(self.conv1(x,adj))
        # x = self.bn1(x)
        x1 = self.conv1(x,adj)
        g_score = self.proj(x1)#[nodes_num,1]
        g_score_sorted,sort_idx = torch.sort(g_score,dim=0)
        _,inverse_idx = torch.sort(sort_idx,dim=0)
        sorted_x = g_score_sorted*x1[sort_idx].squeeze()
        sorted_x = torch.transpose(sorted_x,0,1).unsqueeze(0) # [1, dataset.num_classes, num_nodes]
        sorted_x = F.relu(self.conv1d(sorted_x))
        sorted_x = F.dropout(sorted_x,p=0.5,training=self.training)
        sorted_x = self.conv1d_2(sorted_x)
        sorted_x = torch.transpose(sorted_x.squeeze(),0,1)# [num_nodes, dataset.num_classes]
        x2 = sorted_x[inverse_idx].squeeze()  # [num_nodes, dataset.num_classes]

        out = torch.cat([x1,x2],dim=1)
        out = F.dropout(out,p=0.5,training=self.training)
        out = self.lin(out)
        
        GCN_result = torch.matmul(self.Q,out)
        return GCN_result
        # return F.softmax(GCN_result,dim=1)
'''

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha

        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)# h.shape: (N, in_features), Wh.shape: (N, out_features)
        # print(Wh.shape)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))#e的维度为N*N
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)#维度为[N,out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes
        #两种repeat的方式
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)#repeat_interleave的参数是重复的次数和维度。,现在的维度是[N*N,out_features]
        Wh_repeated_alternating = Wh.repeat(N, 1)#现在的维度是[N*N,out_features]
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)#按列拼接在一起
        return all_combinations_matrix.view(N, N, 2 * self.out_features)#现在的维度是[N,N,2*out_features]

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads,Q):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        self.bn = nn.BatchNorm1d(num_features=nfeat)
        self.bn1 = nn.BatchNorm1d(num_features=nout)
        self.bn2 = nn.BatchNorm1d(num_features=nheads*nhid)
        self.act1 = nn.LeakyReLU()
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)
        self.Q = Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q,[21025,196]
        # self.softmax_linear = nn.Sequential(nn.Linear(64,16))
    def forward(self, x,adj):
        (h,w,b)=x.shape
        x = x.reshape([h*w,-1])
        print(x.shape)
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = self.act1(self.bn1(x))
        x = torch.matmul(self.Q, x)
        return x
'''
class NL_GAT(nn.Module):
    def __init__(self,nfeat, nhid, nout, dropout, alpha, nheads , Q , class_num):
        super(NL_GAT,self).__init__()
        self.conv1 = GAT(nfeat=nfeat,nhid=nhid,nout=nout,dropout=dropout,alpha=alpha,nheads= nheads)
        # self.proj = nn.Linear(16,1)
        self.proj = nn.Linear(nout,1)
        self.kernel = 5
        self.conv1d = nn.Conv1d(nout,nout,self.kernel,padding=int((self.kernel-1)/2))
        # self.conv1d = nn.Conv1d(16, 16, self.kernel, padding=int((self.kernel - 1) / 2))
        self.conv1d_2 = nn.Conv1d(nout, nout, self.kernel, padding=int((self.kernel - 1) / 2))
        # self.conv1d_2 = nn.Conv1d(16, 16, self.kernel, padding=int((self.kernel - 1) / 2))
        self.Q = Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q,[21025,196]
        self.lin = nn.Linear(2*nout,class_num)
        # self.lin = nn.Linear(2 * 16, 16)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.proj.reset_parameters()
        self.conv1d.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x,A):
        (h,w,b) = x.shape
        re_x = x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), re_x)
        x1 = F.relu(self.conv1(superpixels_flatten,A))

        g_score = self.proj(x1)  # [num_nodes, 1]
        g_score_sorted, sort_idx = torch.sort(g_score, dim=0)
        _, inverse_idx = torch.sort(sort_idx, dim=0)

        sorted_x = g_score_sorted * x1[sort_idx].squeeze()
        sorted_x = torch.transpose(sorted_x, 0, 1).unsqueeze(0)  # [1, dataset.num_classes, num_nodes]
        sorted_x = F.relu(self.conv1d(sorted_x))
        sorted_x = F.dropout(sorted_x, p=0.5, training=self.training)
        sorted_x = self.conv1d_2(sorted_x)
        sorted_x = torch.transpose(sorted_x.squeeze(), 0, 1)  # [num_nodes, dataset.num_classes]
        x2 = sorted_x[inverse_idx].squeeze()  # [num_nodes, dataset.num_classes]

        out = torch.cat([x1, x2], dim=1)
        out = self.lin(out)
        GAT_result = torch.matmul(self.Q, out)
        return GAT_result
        # return F.softmax(GAT_result, dim=1)

class Denoise(nn.Module):
    def __init__(self,channel:int,layers:int):
        super(Denoise,self).__init__()
        self.channel = channel
        self.layers = layers
        self.CNN_denoise = nn.Sequential()
        for i in range(self.layers):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel,128,kernel_size=(1,1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
    def forward(self,x):
        return self.CNN_denoise(x)

class NL_Combined_GCN_GAT(nn.Module):
    def __init__(self,nfeat, nhid, nout, dropout, alpha, nheads , Q , class_num):
        super(NL_Combined_GCN_GAT,self).__init__()
        self.conv1 = AF2GNN(in_channels=nfeat,nhid=nhid,out_channels=nout,dropout=dropout,alpha=alpha,nheads=nheads,filter_num=2,class_num=class_num,Q=Q)
        self.proj = nn.Linear(nout,1)
        self.kernel = 5
        self.conv1d = nn.Conv1d(nout,nout,self.kernel,padding=int((self.kernel-1)/2))
        self.conv1d_2 = nn.Conv1d(nout, nout, self.kernel, padding=int((self.kernel - 1) / 2))
        self.Q = Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q,[21025,196]
        self.lin = nn.Sequential(nn.Linear(2*nout,class_num),nn.BatchNorm1d(class_num))
        self.bn = nn.BatchNorm1d(2*nout)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.proj.reset_parameters()
        self.conv1d.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x,A):
        (h,w,b) = x.shape
        re_x = x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), re_x)
        x1 = self.conv1(superpixels_flatten, A)
        g_score = self.proj(x1)  # [num_nodes, 1]
        g_score_sorted, sort_idx = torch.sort(g_score, dim=0)#数值和下标
        _, inverse_idx = torch.sort(sort_idx, dim=0)

        sorted_x = g_score_sorted * x1[sort_idx].squeeze()
        sorted_x = torch.transpose(sorted_x, 0, 1).unsqueeze(0)  # [1, dataset.num_classes, num_nodes]
        sorted_x = F.relu(self.conv1d(sorted_x))
        sorted_x = F.dropout(sorted_x, p=0.5, training=self.training)
        sorted_x = self.conv1d_2(sorted_x)
        sorted_x = torch.transpose(sorted_x.squeeze(), 0, 1)  # [num_nodes, dataset.num_classes]
        x2 = sorted_x[inverse_idx].squeeze()  # [num_nodes, dataset.num_classes]

        out = torch.cat([x1, x2], dim=1)
        out = self.bn(out)
        out = self.lin(out)
        GAT_result = torch.matmul(self.Q, out)
        return F.softmax(GAT_result, dim=1)
'''