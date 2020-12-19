import torch as th
import torch.nn.functional as F


# 后面也许可以把后三个参数合成一个
def BPRLoss(weight_decay, part_users, pos_items, neg_items, prepare, emb_users_out, emb_items_out):
    emb_part_users_out, emb_pos_out, emb_neg_out, \
    emb_part_users, emb_pos, emb_neg = get_embedding(part_users, pos_items, neg_items,
                                                     prepare, emb_users_out, emb_items_out)

    reg_loss = (1 / 2) * (emb_part_users.norm(2).pow(2) +
                          emb_pos.norm(2).pow(2) +
                          emb_neg.norm(2).pow(2)) / len(emb_users_out)

    pos_score = th.mul(emb_users_out, emb_pos_out)
    pos_score = th.sum(pos_score, dim=1)
    neg_score = th.mul(emb_users_out, emb_neg_out)
    neg_score = th.sum(neg_score, dim=1)

    loss = th.mean(F.softplus(neg_score - pos_score))

    bpr_loss = loss + weight_decay * reg_loss

    return bpr_loss


def compute_rating(part_users, emb_users_out, emb_items_out):
    emb_users_out = emb_users_out[part_users]
    rating = F.sigmoid(th.matmul(emb_users_out, emb_items_out.t()))
    return rating


def get_embedding(part_users, pos_items, neg_items, prepare, emb_users_out, emb_items_out):
    emb_users, emb_items = prepare.emb_user, prepare.emb_item

    emb_part_users_out = emb_users_out[part_users]
    emb_pos_out = emb_items_out[pos_items]
    emb_neg_out = emb_items_out[neg_items]

    emb_part_users = emb_users[part_users]
    emb_pos = emb_items[pos_items]
    emb_neg = emb_items[neg_items]

    return emb_part_users_out, emb_pos_out, emb_neg_out, emb_part_users, emb_pos, emb_neg
