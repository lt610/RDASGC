import numpy as np
from train.dataset import Dataset


def uniform_sample(dataset: Dataset):
    n_users, n_relation, n_items, all_pos = dataset.n_users, dataset.train_data_size, dataset.n_items, dataset.all_pos
    sample_users = np.random.randint(0, n_users, n_relation)
    sample_result = []
    for i, user in enumerate(sample_users):
        pos_items = all_pos[user]
        if len(pos_items) == 0:
            continue
        sample_pos_id = np.random.randint(0, len(pos_items))
        pos_item = pos_items[sample_pos_id]
        while True:
            neg_item = np.random.randint(0, n_items)
            if neg_item in pos_items:
                continue
            else:
                break
        sample_result.append([user, pos_item, neg_item])
    return np.array(sample_result)


# ？原作者是在to(device)之后shuffle的呀，会不会有什么影响，后续改进一下
def shuffle(users, pos, neg):
    indexs = np.arange(len(users))
    np.random.shuffle(indexs)
    return users[indexs], pos[indexs], neg[indexs]


def minibatch(batch_size, *tensors):
    if len(tensors) == 1:
        for i in range(0, len(tensors[0]), batch_size):
            yield tensors[0][i: i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            # ?tuple(a, b, c)为啥就会报错，而且生成的那个啥用来索引embedding也不行
            yield tuple(x[i:i + batch_size] for x in tensors)
