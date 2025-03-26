import os
import pandas as pd

files_here = os.listdir(".")
csv_list = [file_name for file_name in files_here if file_name.endswith(".csv")]
concat_list = []
for csvFile in csv_list:
    print(csvFile)
    df = pd.read_csv(csvFile)
    concat_list.append(df)
concat_df = pd.concat(concat_list,join='inner')
concat_df.to_csv("concat.csv", index=False)