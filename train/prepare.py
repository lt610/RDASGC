import torch.nn as nn
import torch as th
from train.dataset import Dataset
from net.rdasgc_net import RDASGCNet


class Prepare(object):
    def __init__(self, device, params, model_name):
        self.device = device
        self.params = params
        self.model_name = model_name

        self.graph, self.test_data, self.dataset = None, None, None
        self.emb_user, self.emb_item, self.emb_feat = None, None, None
        self.model, self.optimizer = None, None

    def prepare_data(self):
        dataset = Dataset(dataset=self.params["dataset"])
        graph = dataset.get_dgl_graph().to(self.device)
        test_data = dataset.test_dict
        self.graph, self.test_data, self.dataset = graph, test_data, dataset
        return graph, test_data, dataset

    def prepare_embedding(self, n_user, n_item):
        emb_user = nn.Embedding(num_embeddings=n_user, embedding_dim=self.params["emb_dim"]).to(self.device)
        emb_item = nn.Embedding(num_embeddings=n_item, embedding_dim=self.params["emb_dim"]).to(self.device)
        emb_features = th.cat([emb_user.weight, emb_item.weight])
        self.emb_user, self.emb_item, self.emb_feat = emb_user, emb_item, emb_features
        return emb_user, emb_item, emb_features

    def prepare_model(self, emb_features):
        if self.model_name == "rdasgc":
            model = RDASGCNet(k=self.params["k"])
        else:
            pass
        model = model.to(self.device)
        optimizer = th.optim.Adam([{"params": model.parameters()},
                                   {"params": emb_features}],
                                  lr=self.params["lr"])
        self.model, self.optimizer = model, optimizer

        return model, optimizer
