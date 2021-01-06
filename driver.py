import random
import time
from subprocess import Popen , DEVNULL
from sacred import Experiment
import os
import itertools
import multiprocessing
from multiprocessing import Process
from train.train import exec_cmds, exec_cmd, parallel_exec_cmds

ex = Experiment()

@ex.config
def base_config():
    tags = "debug"
    config_name = "rsgc"
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
def main(gpus, max_proc_num, parallel_proc_num, wait_time, seed,
         tags, config_name, model_name, params, ex_name):

    prefix = 'python main.py --name {} with "gpus={}" max_proc_num={} seed={}' \
             ' tags={} config_name={} model_name={}'.format(ex_name, gpus, max_proc_num,
                                             seed, tags, config_name, model_name)
    # suffix = ">/dev/null 2>&1 &"
    suffix = ""
    templete = '{} "params={}" {}'

    if tags == "debug":
        cmd = templete.format(prefix, params, suffix)
        exec_cmd(cmd)

    elif tags == "final":
        cmds = []
        for p in params.values():
            cmd = templete.format(prefix, p, suffix)
            cmds.append(cmd)

        random.shuffle(cmds)
        parallel_exec_cmds(parallel_proc_num=parallel_proc_num, wait_time=wait_time, cmds=cmds)

    elif tags in ["search", "analyze"]:
        keys = list(params.keys())
        values = list(params.values())
        p = {}
        cmds = []
        n = len(keys)
        ps = eval("itertools.product({})".format(", ".join(["values[{}]".format(i) for i in range(n)])))

        for t in ps:
            for i in range(n):
                p[keys[i]] = t[i]
            cmd = templete.format(prefix, p, suffix)
            cmds.append(cmd)

        random.shuffle(cmds)
        parallel_exec_cmds(parallel_proc_num=parallel_proc_num, wait_time=wait_time, cmds=cmds)



