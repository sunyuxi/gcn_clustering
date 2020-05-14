###################################################################
# File Name: gcn.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri 07 Sep 2018 01:16:31 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, feature_size, dropout=True, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.feature_size = feature_size
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(self.feature_size, self.feature_size)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*self.feature_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):

        neighbor_size = input.size()[1]
        h = torch.mm(input.view(-1, self.feature_size), self.W)
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, self.feature_size), h.repeat(1, N, 1).view(-1, self.feature_size)], dim=1).view(-1, 2*self.feature_size)
        e = F.relu(torch.mm(a_input, self.a).view(-1, neighbor_size, neighbor_size))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.bmm(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()
    def forward(self, features, A ):
        x = torch.bmm(A, features)
        return x 

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(
                torch.FloatTensor(in_dim *2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg

    def forward(self, features, A):
        b, n, d = features.shape
        assert(d==self.in_dim)
        agg_feats = self.agg(features,A)
        cat_feats = torch.cat([features, agg_feats], dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out 
        

class gcn(nn.Module):
    def __init__(self, agg_type):
        super(gcn, self).__init__()
        self.bn0 = nn.BatchNorm1d(512, affine=False)
        if agg_type == 1:
            self.conv1 = GraphConv(512, 512, MeanAggregator)
            self.conv2 = GraphConv(512, 512, MeanAggregator)
            self.conv3 = GraphConv(512, 256, MeanAggregator)
            self.conv4 = GraphConv(256, 256,MeanAggregator)
        elif agg_type == 2:
            self.conv1 = GraphConv(512, 512, GraphAttentionLayer(512))
            self.conv2 = GraphConv(512, 512, GraphAttentionLayer(512))
            self.conv3 = GraphConv(512, 256, GraphAttentionLayer(512))
            self.conv4 = GraphConv(256, 256, GraphAttentionLayer(512))
        else:
            print('error aggregator type')
            assert(False)
        
        self.classifier = nn.Sequential(
                            nn.Linear(256, 256),
                            nn.PReLU(256),
                            nn.Linear(256, 2))
    
    def forward(self, x, A, one_hop_idcs, train=True):
        # data normalization l2 -> bn
        B,N,D = x.shape
        #xnorm = x.norm(2,2,keepdim=True) + 1e-8
        #xnorm = xnorm.expand_as(x)
        #x = x.div(xnorm)
        
        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B,N,D)


        x = self.conv1(x,A)
        x = self.conv2(x,A)
        x = self.conv3(x,A)
        x = self.conv4(x,A)
        k1 = one_hop_idcs.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B,k1,dout).cuda()
        for b in range(B):
            edge_feat[b,:,:] = x[b, one_hop_idcs[b]]  
        edge_feat = edge_feat.view(-1,dout)
        pred = self.classifier(edge_feat)
            
        # shape: (B*k1)x2
        return pred





