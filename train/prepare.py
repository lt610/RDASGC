import torch.nn as nn
import torch as th

from net.rappnp_net import RAPPNPNet
from net.rdagnn_net import RDAGNNNet
from train.dataset import Dataset
from net.rsgc_net import RSGCNet


class Prepare(object):
    def __init__(self, device, params, model_name):
        self.device = device
        self.params = params
        self.model_name = model_name

        self.graph, self.test_dict, self.dataset = None, None, None
        self.n_users, self.n_items = None, None
        self.emb_users_ini, self.emb_items_ini, self.emb_features = None, None, None
        self.model, self.optimizer = None, None

    def prepare_data(self):
        dataset = Dataset(dataset=self.params["dataset"])
        graph = dataset.get_dgl_graph().to(self.device)
        test_dict = dataset.test_dict
        self.graph, self.test_dict, self.dataset = graph, test_dict, dataset
        self.n_users, self.n_items = dataset.n_users, dataset.n_items
        return graph, test_dict, dataset

    def prepare_embedding(self, n_user, n_item):
        emb_users_ini = nn.Embedding(num_embeddings=n_user, embedding_dim=self.params["emb_dim"]).to(self.device)
        emb_items_ini = nn.Embedding(num_embeddings=n_item, embedding_dim=self.params["emb_dim"]).to(self.device)

        nn.init.normal_(emb_users_ini.weight, std=0.1)
        nn.init.normal_(emb_items_ini.weight, std=0.1)

        self.emb_users_ini, self.emb_items_ini = emb_users_ini, emb_items_ini
        return emb_users_ini, emb_items_ini

    def prepare_model(self, emb_users_ini, emb_items_ini):
        if self.model_name == "rsgc":
            model = RSGCNet(
                k=self.params["k"],
                aggr=self.params["aggr"]
            )
        elif self.model_name == "rdagnn":
            model = RDAGNNNet(
                out_dim=self.params["emb_dim"],
                k=self.params["k"]
            )
        elif self.model_name == "rappnp":
            model = RAPPNPNet(
                k=self.params["k"],
                alpha=self.params["alpha"]
            )
        else:
            pass
        model = model.to(self.device)
        optimizer = th.optim.Adam([{"params": model.parameters(), "weight_decay": self.params["param_regular"]},
                                   {"params": emb_users_ini.parameters()},
                                   {"params": emb_items_ini.parameters()}],
                                  lr=self.params["lr"])

        self.model, self.optimizer = model, optimizer

        return model, optimizer

    def prepare_features(self, emb_users_ini, emb_items_ini):
        emb_features = th.cat([emb_users_ini.weight, emb_items_ini.weight])
        self.emb_features = emb_features
        return emb_features
