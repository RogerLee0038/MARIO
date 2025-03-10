import tomli
import tomli_w
import pickle
import numpy as np
import os
import subprocess
import pandas as pd
# from nevergrad.benchmark.utils import Selector
def create_seed_generator(seed):
    generator = None if seed is None else np.random.RandomState(seed=seed)
    while True:
        yield None if generator is None else generator.randint(2**32)
with open("confxp.toml", "rb") as f:
    xp_dict = tomli.load(f)
exp_name = xp_dict["exp_name"]
exp_seed = xp_dict["exp_seed"]
func_sleep = xp_dict["func_sleep"]
func_names = xp_dict["func_names"]
seedg = create_seed_generator(exp_seed)
# optims = sorted(
#     x for x, y in optimizers.registry.items() if y.one_shot and "arg" not in x and "mal" not in x
# )
algorithms = xp_dict["algorithms"]
budget_factors = xp_dict["budget_factors"]
num_worker_lists = xp_dict["num_worker_lists"]
togethers = xp_dict["togethers"]
repetition= xp_dict["repetition"]

conf_dicts = []
for _ in range(repetition): # With next(seedg)
    conf_dicts += [
        dict(
            algorithm = dict(algo_list=algorithm, num_worker_list=num_worker_list, together=together, seed=next(seedg)),
            target = dict(func_name="My" + func_name, func_sleep=func_sleep),
            strategy = dict(budget_factor=budget_factor, pass_percent=0.3, pass_divide=5, update_num_workers=True, using_portfolio=True)
            )
        for algorithm, num_worker_list in zip(algorithms, num_worker_lists)
        for together in togethers[0:1]
        for func_name in func_names
        for budget_factor in budget_factors
    ]
    # Prevent len(algorithm) == 1 cases from being repeated
    conf_dicts += [
        dict(
            algorithm = dict(algo_list=algorithm, num_worker_list=num_worker_list, together=together, seed=next(seedg)),
            target = dict(func_name="My" + func_name, func_sleep=func_sleep),
            strategy = dict(budget_factor=budget_factor, pass_percent=0.3, pass_divide=5, update_num_workers=True, using_portfolio=True)
            )
        for algorithm, num_worker_list in zip(algorithms, num_worker_lists)
        for together in togethers[1:]
        for func_name in func_names
        for budget_factor in budget_factors
        if not len(algorithm) == 1
    ]

for index, conf in enumerate(conf_dicts):
    with open("conf.toml", "wb") as f1:
        tomli_w.dump(conf, f1)
    proc = subprocess.Popen(["python3", "runone.py"])
    print("waiting for runone.py ...")
    proc.wait()
    algo_list = conf["algorithm"]["algo_list"] 
    num_worker_list = conf["algorithm"]["num_worker_list"] 
    together = conf["algorithm"]["together"] 
    func_name = conf['target']["func_name"][2:]
    budget_factor = conf['strategy']["budget_factor"]
    if len(algo_list) == 1:
        algo_name = algo_list[0]
    elif not together:
        algo_name = "Or".join(algo_list)
    else:
        algo_name = "Plus".join(algo_list)
    num_worker_name = "num_worker"+"_".join([str(num) for num in num_worker_list])
    target_dir = os.path.join(
        "alldatas", algo_name, num_worker_name, func_name, "{}".format(budget_factor)
        )
    os.makedirs(target_dir, exist_ok=True)
    subprocess.call(["mv", "results", "{}/{}".format(target_dir, index)])
subprocess.call(["python3", "get_csv.py"])
subprocess.call(["mv", "temp.csv", "{}_po.csv".format(exp_name)])
