"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""
import dgl
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

# 自定义节点更新特征的方式，这里是mlp+bn+relu，实际是对应原文公式4.1第一项
class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""

    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h

class MLP(nn.Module):
    """MLP with linear output"""

    # num_layers:共有多少层
    # input_dim：输入维度
    # hidden_dim：隐藏层维度，所有隐藏层维度都一样
    # hidden_dim：输出维度
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model这个时候只有一层MLP
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 层数合法性判断
        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:  # 只有一层则按线性变换来玩，输入就是输出
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:  # 有多层则按下面代码处理
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))  # 第一层比较特殊，输入维度到隐藏层维度
            for layer in range(num_layers - 2):  # 中间隐藏层可以循环来玩，隐藏层维度到隐藏层维度
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))  # 最后一层，隐藏层维度到输出维度

            for layer in range(num_layers - 1):  # 除了最后一层都加BN
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):  # 前向传播
        if self.linear_or_not:  # 只有单层MLP
            # If linear model
            return self.linear(x)
        else:  # 多层MLP
            # If MLP
            h = x
            for i in range(self.num_layers - 1):  # 除最后一层外都加一个relu
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)  # 最后一层用线性变换把维度转到输出维度

class GIN(nn.Module):
    """GIN model初始化"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type,device,Q):
        """

        :param num_layers: 5
        :param num_mlp_layers: 2
        :param input_dim: 200
        :param hidden_dim:128
        :param output_dim:16
        :param final_dropout:0.5
        :param learn_eps:True
        :param graph_pooling_type:["sum", "mean", "max"]
        :param neighbor_pooling_type:["sum", "mean", "max"]
        """
        """model parameters setting
        Paramters
        ---------
        num_layers: int这个是GIN的层数
            The number of linear layers in the neural network
        num_mlp_layers: intMLP的层数
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float最后一层的抓爆率
            dropout ratio on the final linear layer
        learn_eps: boolean在学习epsilon参数时是否区分节点本身和邻居节点
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str邻居汇聚方式，原文公式4.1的后半部分
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str全图汇聚方式，和上面的邻居汇聚方式可以不一样
            how to aggregate entire nodes in a graph (sum, mean or max)
        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.device = device
        self.Q = Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q,[21025,196]
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):  # GIN有几层，除了最后一层每层都定义一个MLP（num_mlp_layers层）来进行COMBINE
            if layer == 0:  # 第一层GIN，注意输入维度，
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            # 更新特征的方式是ApplyNodeFunc，邻居汇聚方式为neighbor_pooling_type
            # 具体参考：https://docs.dgl.ai/api/python/nn.pytorch.html#ginconv
            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        # 以下代码是将每一层点的表征保存下来，然后作为最后的图的表征计算
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        # 图表征消息汇聚的方式
        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):  # 前向传播
        (a,b,c) = g.shape
        re_g = g.reshape([a*b, -1])
        # print(re_g.shape)
        g = torch.mm(self.norm_col_Q.t(), re_g)
        # g = dgl.graph(g.cpu())
        # g = g.to(device=self.device)
        # h = h.to(device=self.device)
        # print(g.device)
        # print(h.device)
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]
        for i in range(self.num_layers - 1):  # 根据GIN层数做循环
            h = self.ginlayers[i](g, h)  # 做原文公式4.1的操作
            h = self.batch_norms[i](h)  # 接BN
            h = F.relu(h)  # 接RELU
            hidden_rep.append(h)  # 保存每一层的输出，作为最后图表征的计算
        score_over_layer = 0
        # 根据hidden_rep计算图表征
        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        print(score_over_layer.shape)
        return score_over_layer