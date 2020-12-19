import numpy as np


class Metric(object):
    def __init__(self, test_data, pred_data, topk):
        self.test_data = test_data
        self.pred_data = pred_data
        self.topk = topk

    def recall_and_precis_atk(self):
        right_pred = self.pred_data[:, :self.topk].sum(1)

        precis_n = self.topk
        recall_n = np.array([len(t) for t in self.test_data])

        precis = np.sum(right_pred)/precis_n
        recall = np.sum(right_pred/recall_n)

        return precis, recall

    def ndgc_atk(self):
        pre_data = self.pred_data[:, :self.topk]

        ideal_result = np.zeros((len(pre_data), k))
        for i, items in enumerate(self.test_data):
            n = min(self.topk, len(items))
            ideal_result[i, :n] = 1
        idcg = np.sum(ideal_result * 1. / np.log2(np.arange(2, self.topk + 2)))
        idcg[idcg == 0.] = 1.

        dcg = pre_data * (1. / np.log2(np.arange(2, self.topk + 2)))

        ndcg = dcg / idcg
        ndcg[np.isnan(ndcg)] = 0.
        return np.sum(ndcg)

