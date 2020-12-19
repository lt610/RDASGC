import torch.nn as nn
import torch as th
from train.dataset import Dataset
from net.rdasgc_net import RDASGCNet


class Prepare(object):
    def __init__(self, device, params, model_name):
        self.device = device
        self.params = params
        self.model_name = model_name

    def prepare_data(self):
        dataset = Dataset(dataset=self.params["dataset"])
        graph = dataset.get_dgl_graph().to(self.device)
        test_data = dataset.test_dict
        n_user, n_item = dataset.n_users, dataset.n_items

        return graph, test_data, n_user, n_item, dataset

    def prepare_embedding(self, n_user, n_item):
        emb_user = nn.Embedding(num_embeddings=n_user, embedding_dim=self.params["emb_dim"]).to(self.device)
        emb_item = nn.Embedding(num_embeddings=n_item, embedding_dim=self.params["emb_dim"]).to(self.device)
        emb_features = th.cat([emb_user.weight, emb_item.weight])
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
        return model, optimizer
