import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import tomli

#plt.style.use('_mpl-gallery-nogrid')


def get_stacked_workers(loc, algo_list, data_files, fb):
    algo_workers = dict.fromkeys(algo_list,None)
    for algo_name, data_file in zip(algo_list, data_files):
        with open(os.path.join(loc, data_file),"rb") as handler:
            database = pickle.load(handler)
        for i in range(fb):
            temp_workers = [1 for record in database if record['number'] == i]
            if algo_workers[algo_name] is None:
                algo_workers[algo_name] = [len(temp_workers)]
            else:
                algo_workers[algo_name].append(len(temp_workers))
    return np.array(list(algo_workers.values()))


# make data
with open("conf.toml", "rb") as f:
    toml_dict = tomli.load(f)
algo_list = toml_dict["algorithm"]["algo_list"]
num_worker_list = toml_dict["algorithm"]["num_worker_list"]
unique_algo_list = ["{}_{}".format(index, algo) for index, algo in enumerate(algo_list)]
data_files = ["database_{}_{}.pkl".format(unique_algo, num_worker) 
              for unique_algo, num_worker in zip(unique_algo_list, num_worker_list)]
in_dim = toml_dict['target'].get('in_dim')
assert in_dim is not None, "set the in_dim by hand!"
budget_factor = toml_dict['strategy']["budget_factor"]
fb = budget_factor * (in_dim + 1)
x = np.arange(1, fb+1)
y = get_stacked_workers('.', algo_list, data_files, fb)

fig, ax = plt.subplots(figsize=(12,9), layout="constrained")
for index, row in enumerate(y):
    ax.bar(
        x, row, bottom=y[:index].sum(axis=0),label="{}".format(algo_list[index]),
        width=1, align='center', edgecolor='white'
        )

ax.set(xlabel = "Folded Budget", xlim=(0, fb), # x_ticks = arange(1, fb+1)
       ylabel = "Distribution of Workers")
ax.legend()
plt.savefig("workers.png")
