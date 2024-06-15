import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GCN2Conv

import scipy.sparse as sp
from matplotlib import pyplot as plt
import pdb
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HGDNN(nn.Module):

    def __init__(self, num_edge, w_in,num_node, w_out, num_class, dropout, args=None):
        super(HGDNN, self).__init__()
        # 把定义的数值传过来
        self.num_edge = num_edge
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.dropout = dropout
        self.args = args
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(1, self.w_out), requires_grad=True)
        self.num_node = num_node
        self.attw = nn.Parameter(torch.Tensor(1, self.num_node))
        self.attb = nn.Parameter(torch.Tensor(1, self.w_out))
        self.attq = nn.Parameter(torch.Tensor(1, self.w_out))
        self.DGCN = DeepGraphConv(self.w_in + 5 * self.w_out, hidden_channels=self.w_out,dropout=self.dropout)
        self.loss = nn.CrossEntropyLoss()
        self.register_buffer("epsilon", torch.FloatTensor([1e-12]))
        # 线性层
        self.linearg = nn.Linear(self.w_in, self.w_in)
        self.linear1 = nn.Linear(self.w_out, self.w_out)#654, 64, 64, self.w_out
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attw)
        nn.init.xavier_uniform_(self.attq)
        #self.bias.data.fill_(0.5)
        #nn.init.xavier_uniform_(self.bias)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.attb)
    # Relation-Aware-gcn
    def gcn_conv(self, X, H):
        X = F.leaky_relu(self.linearg(X))
        X = F.dropout(X, p=0.0, training=self.training)  #ACM=0.2  IMDB=0.3  DBLP=0.0
        X = torch.mm(X, self.weight)
        X = F.leaky_relu(X)
        H = self.norm(H, add=True)
        return torch.mm(H.t(), X)

    # 定义归一化函数
    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.cuda.FloatTensor))
        else:
            H = H * ((torch.eye(H.shape[0]) == 0).type(torch.cuda.FloatTensor)) + torch.eye(H.shape[0]).type(
                torch.cuda.FloatTensor)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv * torch.eye(H.shape[0]).type(torch.cuda.FloatTensor)
        H = torch.mm(deg_inv, H)
        H = H.t()
        return H

    def forward(self, A, X, target_x, target):
        F_X = X
        A = A.permute(2, 0, 1)
        lgcn = []
        watt = torch.FloatTensor([1,1,1,1,1])
        # Relation-Aware GCN
        for i in range(self.num_edge):  # conv的channel数量
            lgcn.append(F.relu(self.gcn_conv(X, A[i])))
            #if i == 0:
            #    X_ = F.relu(self.gcn_conv(X, A[i]))  # X-features; H[i]-第i个channel输出的邻接矩阵A[i]; gcn_conv:Linear
            #else:
            #    X_tmp = F.relu(self.gcn_conv(X, A[i]))
            #    X_ = torch.cat((X_, X_tmp), dim=1)
        #print(self.attq.shape, self.attw.shape,lgcn[0].shape,self.attb.shape)
        for i in range(self.num_edge):
            tm = torch.tanh(torch.mm(self.attw,lgcn[i])+self.attb)
            watt[i] = torch.mm(self.attq, tm.t())
        beta = torch.softmax(watt,dim=0)*self.num_edge
        #beta = torch.FloatTensor([1,1,1,1,1])
        print("beta,watt:",beta, watt)
        #print("self.attw:", self.attw)
        #print("self.attq:", self.attq)
        #print("self.attb:", self.attb)
        for i in range(self.num_edge):
            if i == 0:
                X_ = beta[i]*lgcn[i]
            else:
                X_ = torch.cat((X_, beta[i]*lgcn[i]),dim=1)
        # MeanPooling
        A = torch.mean(A, dim=0)
        X_ = torch.cat((X_, F_X), dim=1)
        
        # DGCN
        #Z = X_
        Z = F.leaky_relu(self.DGCN(X_, A))
        #print(X_.shape, Z.shape)
        # MLP
        Z = self.linear1(Z)
        Z = F.leaky_relu(Z)
        y = self.linear2(Z[target_x])
        loss = self.loss(y, target)
        return loss, y


class DeepGraphConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=64, alpha=0.1, theta=0.5,
                 shared_weights=True, dropout=0.0):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, hidden_channels))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=True))

        self.dropout = dropout
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, adj):
        # x = torch.nn.functional.normalize(x, p=2, dim=1)
        adj = adj.cpu().detach().numpy()
        adj = sp.coo_matrix(adj)
        values = adj.data  # 边上对应权重值weight
        indices = np.vstack((adj.row, adj.col))  # pyG真正需要的coo形式
        edge_index = torch.cuda.LongTensor(indices)  # 我们真正需要的coo形式

        x = F.dropout(x, self.dropout, training=self.training)  # ACM=0.2   IMDB=0.3   DBLP=0.0
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, edge_index)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
