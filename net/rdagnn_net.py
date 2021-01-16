from layer.rdagnn_layer import RDAGNNLayer
import torch.nn as nn
import torch.nn.functional as F
from util.other_util import cal_gain

class RDAGNNNet(nn.Module):
    def __init__(self, out_dim, k=2):
        super(RDAGNNNet, self).__init__()
        self.rdagnn = RDAGNNLayer(out_dim, k)

    def forward(self, graph, features):
        h = self.rdagnn(graph, features)
        return h



# class RDAGNNNet(nn.Module):
#     def __init__(self, out_dim, k=2):
#         super(RDAGNNNet, self).__init__()
#         self.linear1 = nn.Linear(64, 64)
#         self.linear2 = nn.Linear(64, 64)
#         self.activation = F.relu
#         self.dropout = 0
#         self.rdagnn = RDAGNNLayer(out_dim, k)
#
#     def reset_parameters(self):
#         gain = cal_gain(self.activation)
#         nn.init.xavier_uniform_(self.linear1.weight, gain=gain)
#         if self.linear1.bias is not None:
#             nn.init.zeros_(self.linear2.bias)
#         nn.init.xavier_uniform_(self.linear1.weight)
#         if self.linear1.bias is not None:
#             nn.init.zeros_(self.linear2.bias)
#
#     def forward(self, graph, features):
#         h = F.dropout(features, self.dropout, training=self.training)
#         h = self.activation(self.linear1(h))
#         h = F.dropout(h, self.dropout, training=self.training)
#         h = self.linear2(h)
#         h = self.rdagnn(graph, h)
#         return h