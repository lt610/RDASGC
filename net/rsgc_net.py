from layer.rsgc_layer import RSGCLayer
import torch.nn as nn


class RSGCNet(nn.Module):
    def __init__(self, k=2):
        super(RSGCNet, self).__init__()
        self.rsgc = RSGCLayer(k)

    def forward(self, graph, features):
        h = self.rsgc(graph, features)
        return h