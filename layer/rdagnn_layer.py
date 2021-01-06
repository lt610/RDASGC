import torch as th
from torch import nn
from torch.nn import Parameter
import dgl.function as fn
from torch.nn import functional as F

from util.other_util import cal_gain


class RDAGNNLayer(nn.Module):
    def __init__(self, out_dim, k=2):
        super(RDAGNNLayer, self).__init__()
        self.s = Parameter(th.FloatTensor(out_dim, 1))
        self.k = k
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(F.sigmoid)
        nn.init.xavier_normal_(self.s, gain=gain)

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
        H = th.stack(results, dim=1)
        S = F.sigmoid(th.matmul(H, self.s))
        S = S.permute(0, 2, 1)
        H = th.matmul(S, H).squeeze()
        return H