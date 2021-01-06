import multiprocessing

from train.loss import BPRLoss
from util.data_util import uniform_sample
import torch as th
from util.data_util import shuffle, minibatch
from util.emb_util import get_emb_out, split_emb_out, get_emb_ini, compute_rating, tran_one_zero
from train.metric import recall_and_precis_atk, ndgc_atk


# 后面加个full batch GPU版本的
def test(prepare, test_batch_size):
    graph, test_dict, dataset = prepare.graph, prepare.test_dict, prepare.dataset
    emb_users_ini, emb_items_ini, emb_features = prepare.emb_users_ini, prepare.emb_items_ini, prepare.emb_features
    model, optimizer = prepare.model, prepare.optimizer
    topk = prepare.params["topk"]

    multicore = prepare.params["multicore"]
    if multicore > 0:
        pool = multiprocessing.Pool(multicore)

    batches_labels, batches_preds = [], []
    model.eval()
    with th.no_grad():
        test_users = list(test_dict.keys())
        recall, precis, ndcg = 0., 0., 0.
        for batch_users in minibatch(test_batch_size, test_users):
            all_pos = dataset.get_user_pos_items(batch_users)
            batch_labels = [test_dict[u] for u in batch_users]
            batch_users = th.Tensor(batch_users).long().to(prepare.device)

            emb_out = model(graph, emb_features)
            emb_users_out, emb_items_out = split_emb_out(prepare.n_users, prepare.n_items, emb_out)
            rating = compute_rating(batch_users, emb_users_out, emb_items_out)

            exc_idxs, exc_items = [], []
            for i, items in enumerate(all_pos):
                exc_idxs.extend([i] * len(items))
                exc_items.extend(items)
            rating[exc_idxs, exc_items] = -(1<<10)
            _, batch_preds = th.topk(rating, k=topk)
            batch_preds = batch_preds.cpu()

            batches_labels.append(batch_labels)
            batches_preds.append(batch_preds)
    labels_and_preds = zip(batches_labels, batches_preds, [topk for _ in range(len(batches_labels))])
    if multicore > 0:
        results = pool.map(test_one_batch, labels_and_preds)
    else:
        results = [test_one_batch(x) for x in labels_and_preds]

    recall, precis, ndcg = 0., 0., 0.
    for result in results:
        recall += result["recall"]
        precis += result["precis"]
        ndcg += result["ndcg"]
    recall /= len(test_users)
    precis /= len(test_users)
    ndcg /= len(test_users)

    recall = round(recall, 5)
    precis = round(precis, 5)
    ndcg = round(ndcg, 5)

    if multicore > 0:
        pool.close()

    return recall, precis, ndcg


def test_one_batch(labels_and_preds):
    batch_labels = labels_and_preds[0]
    batch_preds = labels_and_preds[1]
    topk = labels_and_preds[2]
    batch_preds = tran_one_zero(batch_labels, batch_preds)
    precis, recall = recall_and_precis_atk(batch_labels, batch_preds, topk)
    ndcg = ndgc_atk(batch_labels, batch_preds, topk)

    result = {
        "precis": precis,
        "recall": recall,
        "ndcg": ndcg
    }
    return result

# def test(prepare, test_batch_size):
#     graph, test_dict, dataset = prepare.graph, prepare.test_dict, prepare.dataset
#     emb_users_ini, emb_items_ini, emb_features = prepare.emb_users_ini, prepare.emb_items_ini, prepare.emb_features
#     model, optimizer = prepare.model, prepare.optimizer
#     topk = prepare.params["topk"]
#
#     model.eval()
#     with th.no_grad():
#         test_users = list(test_dict.keys())
#         recall, precis, ndcg = 0., 0., 0.
#         for batch_users in minibatch(test_batch_size, test_users):
#             all_pos = dataset.get_user_pos_items(batch_users)
#             batch_labels = [test_dict[u] for u in batch_users]
#             batch_users = th.Tensor(batch_users).long().to(prepare.device)
#
#             emb_out = model(graph, emb_features)
#             emb_users_out, emb_items_out = split_emb_out(prepare.n_users, prepare.n_items, emb_out)
#             rating = compute_rating(batch_users, emb_users_out, emb_items_out)
#
#             exc_idxs, exc_items = [], []
#             for i, items in enumerate(all_pos):
#                 exc_idxs.extend([i] * len(items))
#                 exc_items.extend(items)
#             rating[exc_idxs, exc_items] = -(1<<10)
#             _, batch_pred = th.topk(rating, k=topk)
#             batch_pred = batch_pred.cpu()
#             # 放在这儿不好吧，有时间了好好封装一下
#             batch_pred = tran_one_zero(batch_labels, batch_pred)
#             precis_t, recall_t = recall_and_precis_atk(batch_labels, batch_pred, topk)
#             ndcg_t = ndgc_atk(batch_labels, batch_pred, topk)
#             recall += recall_t
#             precis += precis_t
#             ndcg += ndcg_t
#     recall /= len(test_users)
#     precis /= len(test_users)
#     ndcg /= len(test_users)
#     return recall, precis, ndcg