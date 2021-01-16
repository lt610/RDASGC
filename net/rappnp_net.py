import torch.nn as nn
from dgl.nn.pytorch import APPNPConv
import torch.nn.functional as F


# class RAPPNPNet(nn.Module):
#     def __init__(self, k, alpha):
#         super(RAPPNPNet, self).__init__()
#         self.appnp = APPNPConv(k, alpha)
#
#     def forward(self, graph, features):
#         h = self.appnp(graph, features)
#         return h
from util.other_util import cal_gain


class RAPPNPNet(nn.Module):
    def __init__(self, k, alpha):
        super(RAPPNPNet, self).__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 64)
        self.activation = F.relu
        self.dropout = 0
        self.rappnp = APPNPConv(k, alpha)

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_uniform_(self.linear1.weight, gain=gain)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear2.bias)
        nn.init.xavier_uniform_(self.linear1.weight)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, graph, features):
        h = F.dropout(features, self.dropout, training=self.training)
        h = self.activation(self.linear1(h))
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.linear2(h)
        h = self.rappnp(graph, h)
        return h