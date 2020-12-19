from util.data_util import DataLoader

dataloader = DataLoader()
g = dataloader.get_dgl_graph()
print(g)
