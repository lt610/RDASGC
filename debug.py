import multiprocessing

import numpy as np
import torch as th
from torch import nn

from train.dataset import Dataset
from util.data_util import uniform_sample


class Test:
    def __init__(self):
        self.a = 1
        self.p(self)

    def p(self):
        print(self.a)


def test_get_dgl_graph():
    dataloader = DataLoader()
    g = dataloader.get_dgl_graph()
    print(g)


def test_unfixed_params(a, *b):
    print(a)
    for t in b:
        print(t)


def test_range():
    l = [i for i in range(10)]
    for i in range(0, 10, 3):
        print(i)
        print(l[i: i + 3])


def test_out_of_index():
    l = [1, 2, 3]
    print(l[0:10])


def test_zip():
    a = [1, 2, 3]
    b = [4, 5, 6]
    for t1, t2 in zip(a, b):
        print(t1)
        print(t2)


def test_np_add():
    a = np.array([1, 2, 3])
    b = np.array([4, 5])
    print(a + b)


def test_kwargs(a, **info):
    print(a)
    for key, value in info.items():
        print("{}: {}".format(key, value))


def test_embedding():
    emb_users_ini = th.nn.Embedding(num_embeddings=4, embedding_dim=10)
    idx = th.Tensor([[1, 2]]).long()
    print(emb_users_ini)
    print(idx)
    print(emb_users_ini[idx])


def test_uniform_sample():
    dataset = Dataset("gowalla")
    uniform_sample(dataset)




