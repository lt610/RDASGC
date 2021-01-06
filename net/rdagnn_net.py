from layer.rdagnn_layer import RDAGNNLayer
import torch.nn as nn


class RDAGNNNet(nn.Module):
    def __init__(self, out_dim, k=2):
        super(RDAGNNNet, self).__init__()
        self.rdagnn = RDAGNNLayer(out_dim, k)

    def forward(self, graph, features):
        h = self.rdagnn(graph, features)
        return h


