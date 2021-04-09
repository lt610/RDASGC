import random
import numpy as np
import time
import os
import time
from subprocess import Popen, DEVNULL
from multiprocessing import Process
import numpy as np
import random
import pynvml
from train.loss import BPRLoss
from util.data_util import uniform_sample
import torch as th
from util.data_util import shuffle, minibatch
from util.emb_util import get_emb_out, split_emb_out, get_emb_ini
from sacred import Experiment


def train(prepare, train_batch_size, emb_regular):
    graph, test_dict, dataset = prepare.graph, prepare.test_dict, prepare.dataset
    emb_users_ini, emb_items_ini, emb_features = prepare.emb_users_ini, prepare.emb_items_ini, prepare.emb_features
    model, optimizer = prepare.model, prepare.optimizer

    samples = uniform_sample(dataset)
    # 后期把这里参数优化一下
    sample_users = th.Tensor(samples[:, 0]).long().to(prepare.device)
    sample_pos = th.Tensor(samples[:, 1]).long().to(prepare.device)
    sample_neg = th.Tensor(samples[:, 2]).long().to(prepare.device)
    # print(sample_users)
    sample_users, sample_pos, sample_neg = shuffle(sample_users, sample_pos, sample_neg)
    # print(sample_users)
    n_batch = len(sample_users) // train_batch_size + 1
    avg_loss = 0.

    model.train()
    for (i, (batch_users, batch_pos, batch_neg)) in enumerate(minibatch(train_batch_size, sample_users,
                                                                        sample_pos, sample_neg)):
        emb_out = model(graph, emb_features)
        emb_users_out, emb_items_out = split_emb_out(dataset.n_users, dataset.n_items, emb_out)
        emb_part_users_out, emb_pos_out, emb_neg_out = get_emb_out(batch_users, batch_pos, batch_neg,
                                                                   emb_users_out, emb_items_out)
        emb_part_users_ini, emb_pos_ini, emb_neg_ini = get_emb_ini(batch_users, batch_pos, batch_neg,
                                                                   emb_users_ini, emb_items_ini)

        loss = BPRLoss(emb_regular, emb_part_users_out, emb_pos_out, emb_neg_out,
                       emb_part_users_ini, emb_pos_ini, emb_neg_ini)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss
    avg_loss /= n_batch
    return avg_loss.item()


def log_split(content="-" * 10, n=45):
    print("\n{} {} {}\n".format("-" * n, content, "-" * n))


def log_metric(epoch, **metric):
    info = "Epoch {:04d}".format(epoch)
    for key, value in metric.items():
        info += " | {} {:.5f}".format(key, value)
    print(info)


def rec_metric(ex: Experiment, epoch, **metric):
    for key, value in metric.items():
        ex.log_scalar(key, value, epoch)


def log_rec_metric(ex: Experiment, epoch, metric):
    rec_metric(ex, epoch, **metric)
    log_metric(epoch, **metric)


def generate_random_seeds(seed, nums):
    random.seed(seed)
    return [random.randint(1, 999999999) for _ in range(nums)]


def set_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True


def get_gpu_proc_num(gpu=0, max_proc_num=2):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    process = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    return len(process)


def get_free_gpu(gpus=[0], max_proc_num=2, max_wait=28800):
    waited = 0
    while True:
        for i in range(max_proc_num):
            for gpu in gpus:
                if get_gpu_proc_num(gpu) == i:
                    return gpu
        print("There is no free gpu now. Waiting...")
        time.sleep(300)
        waited += 300
        if waited > max_wait:
            raise Exception("There is no free gpu for {} hours.".format(max_wait // 3600))


def exec_cmd(cmd):
    print("Running cmd: {}".format(cmd))
    proc = Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
    proc.wait()


# 等价于用&&拼接命令行，但是可以多开几个进程运行，从而实现并行化
def exec_cmds(cmds):
    for cmd in cmds:
        exec_cmd(cmd)


def parallel_exec_cmds(parallel_proc_num, wait_time, cmds):
    if parallel_proc_num > len(cmds):
        parallel_proc_num = len(cmds)

    procs = []
    # python list数组不存在越界问题，将来这里可以优化一下代码
    gap = int(len(cmds) / parallel_proc_num + 0.5)
    for i in range(parallel_proc_num):
        start, end = i * gap, min(len(cmds), (i+1)*gap)
        if start >= len(cmds):
            break
        batch_cmds = cmds[start:end]
        procs.append(Process(target=exec_cmds, args=(batch_cmds, )))
    for proc in procs:
        proc.start()
        time.sleep(wait_time)
    for proc in procs:
        proc.join()