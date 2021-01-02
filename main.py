from sacred import Experiment
from sacred.observers import MongoObserver
from train.prepare import Prepare
from train.train import train, generate_random_seeds, set_random_state, get_free_gpu, print_split, log_metric, \
    log_rec_metric
import torch as th
from train.test import test

ex = Experiment()
ex.observers.append(MongoObserver(url='10.192.9.196:27017',
                                      db_name='rdasgc'))

@ex.config
def base_config():
    tags = "debug"
    config_name = "rdasgc"
    ex.add_config("config/base_config/{}.json".format(config_name))
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
        prepare.prepare_embedding(prepare.n_users, prepare.n_items)
        prepare.prepare_model(prepare.emb_users_ini, prepare.emb_items_ini)
        n_log_run = 5
        # 只记录前3 runs的logs
        if run < n_log_run:
            print_split(" {}th run ".format(run))

        set_random_state(random_seeds[run])

        for epoch in range(1, params['num_epochs'] + 1):
            avg_loss = train(prepare, params["train_batch_size"], params["weight_decay"])

            log_rec_metric(ex, epoch, {"avg_loss": avg_loss})
            if epoch % 1 == 0:
                recall, precis, ndcg = test(prepare, params["test_batch_size"])
                metric = {"precis": precis,
                           "recall": recall,
                           "ndcg": ndcg
                           }
                log_rec_metric(ex, epoch, metric)

