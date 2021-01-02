import torch as th
import torch.nn.functional as F
import numpy as np


def compute_rating(part_users, emb_users_out, emb_items_out):
    emb_part_users_out = emb_users_out[part_users]
    rating = F.sigmoid(th.matmul(emb_part_users_out, emb_items_out.t()))
    return rating


def split_emb_out(n_users, n_items, emb_out):
    emb_users_out, emb_items_out = th.split(emb_out, [n_users, n_items])
    return emb_users_out, emb_items_out


def get_emb_out(part_users, pos_items, neg_items, emb_users_out, emb_items_out):
    emb_part_users_out = emb_users_out[part_users]
    emb_pos_out = emb_items_out[pos_items]
    emb_neg_out = emb_items_out[neg_items]
    return emb_part_users_out, emb_pos_out, emb_neg_out


def get_emb_ini(part_users, pos_items, neg_items, emb_users_ini, emb_items_ini):
    # 后续优化一下long()
    emb_part_users_ini = emb_users_ini(part_users)
    emb_pos_ini = emb_items_ini(pos_items)
    emb_neg_ini = emb_items_ini(neg_items)
    return emb_part_users_ini, emb_pos_ini, emb_neg_ini


def tran_one_zero(labels, pred):
    result = []
    for i in range(len(labels)):
        tran = list(map(lambda x: x in labels[i], pred[i]))
        tran = np.array(tran).astype("float")
        result.append(tran)
    return np.array(result).astype("float")