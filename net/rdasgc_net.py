from layer.rdasgc_layer import RADSGCLayer
import torch.nn as nn


class RDASGCNet(nn.Module):
    def __init__(self, k=2):
        super(RDASGCNet, self).__init__()
        self.rdasgc = RADSGCLayer(k)

    def forward(self, graph, features):
        h = self.rdasgc(graph, features)
        return h