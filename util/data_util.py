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
