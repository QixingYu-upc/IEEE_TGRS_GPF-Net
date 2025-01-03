import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
# import NL_GNN

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
        support = torch.matmul(x,self.weight)###N*out_channels
        output = torch.matmul(adj,support)#N*out_channels
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_channels) + '->' + str(self.out_channels) + ')'
class GCN(nn.Module):
    def __init__(self,in_channels,in_hid,classes,dropout=0.5):
        super(GCN,self).__init__()
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
        self.gcn1 = GCN(in_channels=in_channels,in_hid=nhid,classes=in_channels)
        self.gcn2 = GCN(in_channels=in_channels,in_hid=nhid,classes=out_channels)

        # self.gcn1 = GCN(in_channels=in_channels,in_hid = nhid,classes=in_channels)
        # self.gcn2 = GCN(in_channels=in_channels,in_hid=nhid,classes=in_channels)
        # self.gcn3 = GCN(in_channels=in_channels,in_hid=nhid,classes=in_channels)
        # self.gcn4 = GCN(in_channels=in_channels,in_hid=nhid,classes=out_channels)
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.bn2 = nn.BatchNorm1d(num_features=in_channels)
        # self.bn3 = nn.BatchNorm1d(num_features=out_channels)
        self.bn3 = nn.BatchNorm1d(num_features=in_channels)
        self.bn4 = nn.BatchNorm1d(num_features=in_channels)
        self.bn5 = nn.BatchNorm1d(num_features=out_channels)
    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.gcn3.reset_parameters()
        self.gcn4.reset_parameters()
    def forward(self,x,A):
        x = self.bn1(x)
        y = F.leaky_relu(self.bn2(self.gcn1(x, A)))
        x = torch.add(x, y)
        x = F.leaky_relu(self.bn3(self.gcn2(x, A)))
        # x = self.bn1(x)
        # y = F.leaky_relu(self.bn2(self.gcn1(x, A)))
        # x = torch.add(x, y)
        # y = F.leaky_relu(self.bn3(self.gcn2(x, A)))
        # x = torch.add(x,y)
        # y = F.leaky_relu(self.bn4(self.gcn3(x, A)))
        # x = torch.add(x,y)
        # x = F.leaky_relu(self.bn5(self.gcn4(x, A)))
        return x

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
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads):
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
        # self.softmax_linear = nn.Sequential(nn.Linear(64,16))
    def forward(self, x,adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = self.act1(self.bn1(x))
        return x

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
class Res_GCN(nn.Module):
    def __init__(self,in_channels,out_channels,nhid):
        super(Res_GCN, self).__init__()
        self.gcn1 = GCN(in_channels=in_channels,in_hid=nhid,classes=in_channels)
        self.gcn2 = GCN(in_channels=in_channels,in_hid=nhid,classes=out_channels)
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

class AF2GNN1(nn.Module):
    def __init__(self,in_channels,nhid,out_channels,dropout,alpha,nheads,filter_num,class_num,Q):
        super(AF2GNN1, self).__init__()
        self.in_channels = in_channels
        self.nhid = nhid
        self.out_channels = out_channels
        self.dropout = dropout
        self.alpha = alpha
        self.nheads = nheads
        self.filter_num = filter_num
        self.class_num = class_num
        self.conv = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=(1,1))
        self.W = nn.Parameter(torch.FloatTensor(self.filter_num,self.out_channels))
        self.b = nn.Parameter(torch.FloatTensor(self.filter_num,self.out_channels))
        self.W1 = nn.Parameter(torch.FloatTensor(3, self.out_channels))
        self.bb1 = nn.Parameter(torch.FloatTensor(3, self.out_channels))
        self.Q = Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q,[21025,196]
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.bn = nn.BatchNorm1d(num_features=self.in_channels)
        self.bn1 = nn.BatchNorm1d(num_features=self.out_channels)
        self.bn2 = nn.BatchNorm1d(num_features=self.out_channels)
        self.gcn = Res_GCN(in_channels=self.in_channels,out_channels=self.out_channels,nhid=self.nhid)
        self.gat = GAT(nfeat=self.in_channels,nhid=self.nhid,nout=self.out_channels,dropout=self.dropout,alpha=0.2,nheads=self.nheads)
        self.lin = nn.Linear(self.out_channels, self.class_num)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1./math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv,stdv)
        self.b.data.uniform_(-stdv,stdv)
        stdv = 1. / math.sqrt(self.W1.size(1))
        self.W1.data.uniform_(-stdv, stdv)
        self.bb1.data.uniform_(-stdv, stdv)
        self.lin.reset_parameters()
        self.conv.reset_parameters()
    def forward(self,x,adj):
        (h,w,b) = x.shape
        x = x.reshape([h*w,-1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), x)#Transform
        x = self.bn(superpixels_flatten)
        GCN_x = self.gcn(x,adj)
        GAT_x = self.gat(x,adj)#Feature Extraction
        min_x = torch.where(GCN_x>GAT_x,GAT_x,GCN_x)
        max_x = torch.where(GCN_x>GAT_x,GCN_x,GAT_x)
        avg_x = torch.add(torch.mul(GCN_x,self.W[0])+self.b[0],torch.mul(GAT_x,self.W[1])+self.b[1])#Feature Aggregation
        min_x = torch.unsqueeze(min_x,0)
        max_x = torch.unsqueeze(max_x,0)
        avg_x = torch.unsqueeze(avg_x,0)
        x = torch.unsqueeze(torch.cat([min_x,max_x,avg_x],dim=0),dim=0)
        x = F.dropout(x, 0.2 , training=self.training)
        x = self.conv(x+GAT_x+GCN_x)#Feature Update
        x = torch.squeeze(torch.squeeze(x))
        x = self.bn1(x)
        x = F.dropout(x, 0.2 , training=self.training)
        x = self.leakyrelu(x)
        result = torch.matmul(self.Q, x)#Detransform
        return result


class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out
class Channel_only_branch(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax=nn.Softmax(1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x
        return channel_out
class Spatial_only_branch(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()
        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        return spatial_out
def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)
      
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
class CPSPPSELayer(nn.Module):
    def __init__(self,in_channel, channel, reduction=16):
        super(CPSPPSELayer, self).__init__()
        if in_channel != channel:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Sequential(
            nn.Linear(channel*21, channel*21 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*21 // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x) if hasattr(self, 'conv1') else x
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x).view(b, c)  # like resize() in numpy
        y2 = self.avg_pool2(x).view(b, 4 * c)
        y3 = self.avg_pool4(x).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)
        y = self.fc(y)
        b, out_channel = y.size()
        y = y.view(b, out_channel, 1, 1)
        return y
def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

class GlobalContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio=0.5,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(GlobalContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class CNN_Attention(nn.Module):
    def __init__(self,channels,out_channels,layers,class_num):
        super().__init__()
        self.Denoise = Denoise(channels,out_channels,layers)
        # self.att1 = CBAMLayer(out_channels)
        self.c_conv1 = Channel_only_branch(out_channels)
        self.s_conv1 = Spatial_only_branch(out_channels)
        self.ss_conv1 = SSConv(out_channels,out_channels,kernel_size=3)
        self.c_conv2 = Channel_only_branch(out_channels)
        self.s_conv2 = Spatial_only_branch(out_channels)
        self.ss_conv2 = SSConv(out_channels,64,kernel_size=5)
        self.Softmax_linear =nn.Sequential(nn.Linear(64, class_num))

    def forward(self,x,adj):
        (h,w,b) = x.shape
        # print((h,w,b))
        noise = self.Denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        x1 = self.c_conv1(noise)
        x2 = self.s_conv1(noise)
        # x = self.att1(noise)
        x = self.ss_conv1(x1+x2+noise)
        x3 = self.c_conv2(x)
        x4 = self.s_conv2(x3)
        # x = self.att2(x)
        x = self.ss_conv2(x4+noise)
        x = torch.squeeze(x, 0).permute([1, 2, 0])
        x = x.reshape([h * w, -1])        
        return x

class Fusion_GNN_CNN(nn.Module):
    def __init__(self,in_channels,nhid,out_channels,dropout,alpha,nheads,filter_num,class_num,Q):
        super(Fusion_GNN_CNN, self).__init__()
        self.GNN = AF2GNN1(in_channels,nhid,out_channels,dropout,alpha,nheads,filter_num,class_num,Q)
        self.CNN = CNN_Attention(in_channels,int(in_channels*0.98),2,class_num)
        self.Softmax_linear = nn.Sequential(nn.Linear(64,class_num))
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        nn.init.uniform_(self.rate1)
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.bn = nn.BatchNorm1d(64)
        # reset_parameters()
    def reset_parameters(self):
        nn.init.uniform_(self.rate1)
    def forward(self,x,adj):
        GNN_result = self.GNN(x,adj)
        CNN_result = self.CNN(x,adj)
        result = 0.2 * CNN_result + 0.8 * GNN_result
        Y = self.Softmax_linear(result)
        Y = torch.softmax(Y, -1)
        return Y