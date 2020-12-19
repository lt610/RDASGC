from train.loss import BPRLoss
from util.data_util import uniform_sample


def train(prepare, sample):
    graph, test_data, dataset = prepare.graph, prepare.test_data, prepare.dataset
    emb_user, emb_item, emb_feat = prepare.emb_user, prepare.emb_item, prepare.emb_feat
    model, optimizer = prepare.model, prepare.optimizer

    model.train()

    emb_out = model(graph, emb_feat)


    loss = None

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return model

