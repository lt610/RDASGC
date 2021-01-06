import os
from os.path import join
import sys

import dgl
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp


class Dataset(object):
    def __init__(self, dataset="gowalla"):

        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        """
        n_users: 训练集和测试集中所有user的数量，不过训练集和测试集
        一般各自已经包含了所有的user和item，不然有的操作没法进行
        """
        self.n_users = 0
        self.n_items = 0
        path = "data/{}".format(dataset)
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        """
        train_unique_users：N x 1，元素是user id，记录了训练集里所有的user
        train_user：(N*?) x 1，元素是user id；长度(N*?)是为了和train_item对齐
        train_item：(N*?) * 1，元素是item id，记录了训练集里所有user对应的item
        """
        train_unique_users, train_item, train_user = [], [], []
        test_unique_users, test_item, test_user = [], [], []
        """
        train_data_size: 训练集中总的关系（边）数量
        """
        self.train_data_size = 0
        self.test_data_size = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    train_unique_users.append(uid)
                    # 拷贝多份是为了后面构建user-item矩阵用的
                    train_user.extend([uid] * len(items))
                    train_item.extend(items)
                    # max(items)，这代码写的，第一次看漏了把我误导了
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    # traindatasize表示训练集中总的边的数量
                    self.train_data_size += len(items)
        self.train_unique_users = np.array(train_unique_users)
        self.train_user = np.array(train_user)
        self.train_item = np.array(train_item)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if l[1:][0] == '':
                        continue
                    else:
                        items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    test_unique_users.append(uid)
                    test_user.extend([uid] * len(items))
                    test_item.extend(items)
                    # 同上面一样，再重复一边大概是为了防止有些结点在训练集没有出现吧，虽然按道理来说不应该这样
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.test_data_size += len(items)
        # 因为user和item的id都是从0开始的
        self.n_items += 1
        self.n_users += 1
        self.test_unique_users = np.array(test_unique_users)
        self.test_user = np.array(test_user)
        self.test_item = np.array(test_item)

        self.Graph = None

        # (users,items), bipartite graph
        # 这里np.ones(...)标识了非零数据，trainUser标识了行号，trainItem标识了列号，就是对nXm矩阵的一个很简单的压缩
        # 这里是用训练集构建的
        self.user_item_net = csr_matrix((np.ones(len(self.train_user)), (self.train_user, self.train_item)),
                                        shape=(self.n_users, self.n_items))

        """
        all_pos：N x ?的二维数组，元素是item id，记录了训练集里所有user对应的一阶item，作为正例
        test_dict：N x ?的二维数组，元素是item id， 记录了测试集里所有user对应的存在直接关系（一阶）的item
        """
        # pre-calculate
        self.all_pos = self.get_user_pos_items(list(range(self.n_users)))
        self.test_dict = self.build_test()

    def get_sparse_graph(self):
        if self.Graph is None:
            try:
                adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            except:
                # 构建一个(n+m)X(n+m)的邻接矩阵，对应于论文中国的表示
                # ？为什么这么绕，在各个类型的稀疏矩阵间换来换去
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.user_item_net.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.tocsr()
                sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
        return adj_mat

    def get_dgl_graph(self):
        graph = dgl.from_scipy(self.get_sparse_graph())
        return graph

    # 返回的字典数据
    def build_test(self):
        test_data = {}
        for i, item in enumerate(self.test_item):
            user = self.test_user[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    # 这里的positive items是一阶邻居
    # 返回的是一个二维列表
    def get_user_pos_items(self, users):
        pos_items = []
        for user in users:
            pos_items.append(self.user_item_net[user].nonzero()[1])
        return pos_items
