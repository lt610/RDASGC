from sacred import Experiment
from sacred.observers import MongoObserver
from train.prepare import Prepare
from train.train import train, generate_random_seeds, set_random_state, get_free_gpu, log_split, log_metric, \
    log_rec_metric
import torch as th
from train.test import test

ex = Experiment()
ex.observers.append(MongoObserver(url='10.192.9.196:27017',
                                      db_name='rdasgc'))

@ex.config
def base_config():
    tags = "debug"
    config_name = "rdagnn"
    if tags == "debug":
        ex.add_config('config/base_config/{}.json'.format(config_name))
    elif tags == "final":
        ex.add_config("config/best_config/{}.json".format(config_name))
    elif tags == "search":
        ex.add_config("config/search_config/{}.json".format(config_name))
    elif tags == "analyze":
        ex.add_config("config/analyze_config/{}.json".format(config_name))
    else:
        raise Exception("There is no {}".format(tags))
    ex_name = config_name
    model_name = config_name.split("_")[0]

@ex.automain
def main(gpus, max_proc_num, seed, model_name, params):
    # 这里以后如果有需要可以封装到find_free_devices()中
    if not th.cuda.is_available():
        device = "cpu"
    else:
        device = get_free_gpu(gpus, max_proc_num)

    prepare = Prepare(device, params, model_name)
    prepare.prepare_data()

    random_seeds = generate_random_seeds(seed, params["num_runs"])

    for run in range(params["num_runs"]):
        # 一定要放在最前面，确保接下来的所有操作都是可复现的
        set_random_state(random_seeds[run])

        prepare.prepare_embedding(prepare.n_users, prepare.n_items)
        prepare.prepare_model(prepare.emb_users_ini, prepare.emb_items_ini)
        n_log_run = 5
        # 只记录前几个runs的logs
        if run < n_log_run:
            log_split(" {}th run ".format(run + 1))

        counter = 0
        best_score = 0

        for epoch in range(1, params['num_epochs'] + 1):
            prepare.prepare_features(prepare.emb_users_ini, prepare.emb_items_ini)
            avg_loss = train(prepare, params["train_batch_size"], params["emb_regular"])

            log_rec_metric(ex, epoch, {"avg_loss": avg_loss})

            if epoch % 10 == 0:
                print("Test")
                recall, precis, ndcg = test(prepare, params["test_batch_size"])
                metric = {"precis": precis,
                           "recall": recall,
                           "ndcg": ndcg
                           }

                log_rec_metric(ex, epoch, metric)

                # 临时的early stopping，后面有时间加上验证集，现在就算了，不想搞了
                if recall > best_score:
                    best_score = recall
                    counter = 0
                else:
                    counter += 1
                if counter >= 10:
                    break


