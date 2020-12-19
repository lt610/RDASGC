import torch as th
from torch import nn
from torch.nn import Parameter
import dgl.function as fn
from torch.nn import functional as F


class RADSGCLayer(nn.Module):
    def __init__(self, k=2):
        super(RADSGCLayer, self).__init__()
        self.k = k

    def forward(self, graph, features):
        g = graph.local_var()
        h = features
        results = [features]

        # 这里保证度数都是1
        degs = g.in_degrees().float().clamp(min=1)
        norm = th.pow(degs, -0.5)
        norm = norm.to(features.device).unsqueeze(1)

        for _ in range(self.k):
            h = h * norm
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'),
                         fn.sum('m', 'h'))
            h = g.ndata.pop('h')
            h = h * norm
            results.append(h)
        H = th.stack(results, dim=1)
        # 后期可以试一试DAGNN
        H = th.mean(H, dim=1)
        return H