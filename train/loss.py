import torch as th
import torch.nn.functional as F


def BPRLoss(emb_regular, emb_part_users_out, emb_pos_out, emb_neg_out,
            emb_part_users_ini, emb_pos_ini, emb_neg_ini):

    reg_loss = (1 / 2) * (emb_part_users_ini.norm(2).pow(2) +
                          emb_pos_ini.norm(2).pow(2) +
                          emb_neg_ini.norm(2).pow(2)) / len(emb_part_users_out)

    pos_score = th.mul(emb_part_users_out, emb_pos_out)
    pos_score = th.sum(pos_score, dim=1)
    neg_score = th.mul(emb_part_users_out, emb_neg_out)
    neg_score = th.sum(neg_score, dim=1)

    loss = th.mean(F.softplus(neg_score - pos_score))
    # ？加在这里和加在adam优化器里有啥区别呢
    bpr_loss = loss + emb_regular * reg_loss

    return bpr_loss




