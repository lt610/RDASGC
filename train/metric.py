import numpy as np


def recall_and_precis_atk(test_data, pred_data, topk):
    right_pred = pred_data[:, :topk].sum(1)

    precis_n = topk
    recall_n = np.array([len(t) for t in test_data])

    precis = np.sum(right_pred)/precis_n
    recall = np.sum(right_pred/recall_n)

    return precis, recall


def ndgc_atk(test_data, pred_data, topk):
    pre_data = pred_data[:, :topk]

    ideal_result = np.zeros((len(pre_data), topk))
    for i, items in enumerate(test_data):
        n = min(topk, len(items))
        ideal_result[i, :n] = 1
    idcg = np.sum(ideal_result * 1. / np.log2(np.arange(2, topk + 2)), axis=1)
    idcg[idcg == 0.] = 1.

    dcg = pre_data * (1. / np.log2(np.arange(2, topk + 2)))
    dcg = np.sum(dcg, axis=1)

    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

