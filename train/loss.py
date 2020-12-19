import torch as th
import torch.nn.functional as F


# 后面也许可以把后三个参数合成一个
def BPRLoss(weight_deacy, emb_user_out, emb_pos_out, emb_neg_out,
            emb_user_init, emb_pos_init, emb_neg_init):
    reg_loss = (1 / 2) * (emb_user_init.norm(2).pow(2) +
                          emb_pos_init.norm(2).pow(2) +
                          emb_neg_init.norm(2).pow(2)) / len(emb_user_out)

    pos_score = th.mul(emb_user_out, emb_pos_out)
    pos_score = th.sum(pos_score, dim=1)
    neg_score = th.mul(emb_user_out, emb_neg_out)
    neg_score = th.sum(neg_score, dim=1)

    loss = th.mean(F.softplus(neg_score-pos_score))

    bpr_loss = loss + weight_deacy * reg_loss

    return bpr_loss
