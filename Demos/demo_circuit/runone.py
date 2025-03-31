import os
import psutil
import shutil
import time

def copy_files(old_path, new_path):
    file_list = os.listdir(old_path)
    for file_name in file_list:
        full_name = os.path.join(old_path, file_name)
        shutil.copy(full_name, new_path)

PO_SRC = os.getenv("PO_SRC")
if os.path.exists("results"):
    shutil.rmtree("results")
os.makedirs("results")
# copy_files(os.path.join(PO_SRC,"scripts"), "results")
shutil.copy("conf.toml", "results")
cur_dir = os.getcwd()
os.chdir("results")
with open("run.log", "w") as out_file:
    proc = psutil.Popen(["python3", os.path.join(PO_SRC, "scripts_circuit", "run_circuit.py")], stdout = out_file)
    while(True):
        if os.path.exists("finish"):
            print("run_circuit.py done in runone.py")
            proc.terminate()
            break
        else:
            status = proc.status()
            print("in runone.py, run_circuit.py is running, status {}".format(status))
            time.sleep(10)
os.chdir(cur_dir)
# for file_name in os.listdir("workspace"):
#     full_name = os.path.join("workspace", file_name)
#     if not full_name.endswith(".py") and os.path.isfile(full_name):
#         shutil.copy(full_name, "results")
copy_files(os.path.join(PO_SRC,"auxi"), "results")
