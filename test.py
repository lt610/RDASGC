from util.data_util import DataLoader


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




a = Test()

