import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value) 
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out        


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
        Wh = torch.mm(h, self.W)                                 # h.shape: (N, in_features), Wh.shape: (N, out_features)
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
    def __init__(self, nfeat, nhid, adj, nout, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.adj = adj
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return x


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
            padding=kernel_size//2,
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
        self.BN=nn.BatchNorm2d(in_ch)
        
    
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class WFCG(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, model='normal'):
        super(WFCG, self).__init__()
        self.class_count = class_count  # 类别数
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model=model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q,[21025,196]
        # print(self.norm_col_Q.shape)
        layers_count=2

        self.WH = 0
        self.M = 2
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0: 
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
            else:
                # self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                # self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(128),)
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                
                # self.CNN_Branch.add_module('CrissCrossAttention'+str(i), CrissCrossAttention(128))
                # self.CNN_Branch.add_module('CrissCrossAttention'+str(i), CrissCrossAttention(128))
                # self.CNN_Branch.add_module('Attention'+str(i), SKConv(128, self.WH, self.M, 1, 2, stride=1, L=32))
                # self.CNN_Branch.add_module('GCN_Branch'+str(i), ContextBlock(inplanes=128, ratio=1./8., pooling_type='att'))
                # self.CNN_Branch.add_module('attention'+str(i), _NonLocalBlockND(in_channels=128, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True))
                self.CNN_Branch.add_module('Attention'+str(i), PAM_Module(128))
                self.CNN_Branch.add_module('Attention'+str(i), CAM_Module(128))
                self.CNN_Branch.add_module('CNN_Branch'+str(i), SSConv(128, 128,kernel_size=3))
                # self.CNN_Branch.add_module('Drop_Branch'+str(i), nn.Dropout(0.2))
                
            else:
                # self.CNN_Branch.add_module('CrissCrossAttention'+str(i), CrissCrossAttention(128))
                # self.CNN_Branch.add_module('CrissCrossAttention'+str(i), CrissCrossAttention(128))
                # self.CNN_Branch.add_module('Attention'+str(i), SKConv(128, self.WH, self.M, 1, 2, stride=1, L=32))
                # self.CNN_Branch.add_module('GCN_Branch'+str(i), ContextBlock(inplanes=128, ratio=1./8., pooling_type='att'))
                # self.CNN_Branch.add_module('attention'+str(i), _NonLocalBlockND(in_channels=128, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True))
                self.CNN_Branch.add_module('Attention'+str(i), PAM_Module(128))
                self.CNN_Branch.add_module('Attention'+str(i), CAM_Module(128))
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))
        
        self.GAT_Branch = nn.Sequential()
        self.GAT_Branch.add_module('GAT_Branch'+str(i), GAT(nfeat=128, nhid=30, adj=A, nout=64, dropout=0.4, nheads=4, alpha=0.2))

        self.linear1 = nn.Linear(64, 64)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(64)

        self.Softmax_linear =nn.Sequential(nn.Linear(64, self.class_count))

    def forward(self, x: torch.Tensor):#x的维度是[145,145,200]
        (h, w, c) = x.shape
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))#[1,128,145,145]
        noise =torch.squeeze(noise, 0).permute([1, 2, 0])#[145,145,128]
        clean_x=noise#[145,145,128]
        clean_x_flatten=clean_x.reshape([h * w, -1])#[21025,128]
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)#[192,128]
        hx = clean_x

        # CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))
        # CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        H = superpixels_flatten
        H = self.GAT_Branch(H)
        GAT_result = torch.matmul(self.Q, H)
        GAT_result = self.act1(self.bn1(GAT_result))
        Y = GAT_result
        # Y  = 0.05 * CNN_result + 0.95 * GAT_result
        Y = self.Softmax_linear(Y)
        Y = torch.softmax(Y, -1)
        return Y
