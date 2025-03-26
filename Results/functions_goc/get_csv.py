import pickle
import os
import pandas as pd
# from nevergrad.benchmark.utils import Selector

def get_files(path, name):
    file_list = []
    for root, dirs, files in os.walk(path):
        if name in files:
            file_list.append("{}/{}".format(root, name))
    return file_list

dir_name = "functionExps"
csv_name = "concat"
summary_files = get_files(dir_name, "summary.pkl")
summaries = []
for summary_file in summary_files:
    with open(summary_file, "rb") as f2:
        summary = pickle.load(f2)
        summary["dataset_loc"] = os.path.dirname(os.path.abspath(summary_file))
    summaries.append(summary)
df = pd.DataFrame(data=summaries)
for index, row in df.iterrows():
    if "MARIO-1/" in row['dataset_loc']:
        df.loc[index,'optimizer_name'] = "mmine"
    if "MARIO-1se/" in row['dataset_loc']:
        df.loc[index,'optimizer_name'] = "mmine_nu"
df.to_csv("{}.csv".format(csv_name), index=False)
