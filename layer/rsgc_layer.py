import torch as th
from torch import nn
from torch.nn import Parameter
import dgl.function as fn
from torch.nn import functional as F


class RSGCLayer(nn.Module):
    def __init__(self, k=2, aggr="mean"):
        """
        :param k: propagation stps
        :param agrr: layer agrregate方式，有mean和1/k两种
        """
        super(RSGCLayer, self).__init__()
        self.k = k
        self.aggr = aggr

    def forward(self, graph, features):
        """
        :param graph: 只包含边
        :param features: (M+N) X F的embedding，M是item的数量，N是user的数量
        :return: 传播后(M+N) X F的embedding
        """
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
        if self.aggr == "mean":
            H = th.stack(results, dim=1)
            H = th.mean(H, dim=1)
        elif self.aggr == "1/k":
            H = results[0]
            for i in range(1, len(results)):
                emb = results[i] / (i + 1)
                H = H + emb
        elif self.aggr == "none":
            return results[-1]
        return H