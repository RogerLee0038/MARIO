import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('_mpl-gallery-nogrid')

def transform_perform1(p):
    if np.isinf(p):
        return -1
    else:
        tau = 0.5
        if 1-p < tau:
            return (p-0.5)/0.5
        else:
            return 0.

def transform_perform2(p):
    if np.isinf(p):
        return -1
    else:
        tau = 0.05
        if 1-p < tau:
            return 1.
        else:
            return 0.

def transform_perform(p):
    if np.isinf(p):
        return -1
    return p

# after plot_pp
# make data
csv_file = '../concat.csv'
npy_file = './concat.npy'

vectorized_transform = np.vectorize(transform_perform)

df = pd.read_csv(csv_file)
df.head()
optimizer_list = df['optimizer_name'].unique()
func_list = df['func_name'].unique()
print("optimizers:", optimizer_list)
print("func_names:", func_list)

func_results_matrix = np.load(npy_file)
print("func_results_matrix", func_results_matrix.shape)
func_list_temp = func_list
# axis 0 for funcs, axis 1 for optimizers, axis 2 for repetitions, axis 3 for fb_points + 1
temp_perform_matrix = vectorized_transform(func_results_matrix)
best_perform_matrix = np.max(np.mean(temp_perform_matrix, axis=2), axis=-1)
print("best_perform_matrix", best_perform_matrix.shape)

Z = best_perform_matrix.T
Z_func = Z.sum(axis = 0)
Z_opt = Z.sum(axis = 1)
func_map = np.argsort(Z_func)[::-1] # performance high (easy) -> low (tough)
print(func_map)
opt_map = np.argsort(Z_opt)
print(opt_map)

Z = Z[np.ix_(opt_map, func_map)]
func_list_new = ["{}".format(func_list_temp[i][10:-2]) if func_list_temp[i].startswith("MyPutative") else "{}".format(func_list_temp[i][1:-2]) for i in func_map]
func_list_new[func_list_new.index('two_stage_opamp')] = 'TwoStageOpamp'
func_list_new[func_list_new.index('chargepump')] = 'ChargePump'
func_list_new[func_list_new.index('ota')] = 'OTA'
func_list_new[func_list_new.index('lna')] = 'LNA'
func_list_new[func_list_new.index('class_e')] = 'ClassE'
func_list_new[func_list_new.index('gainboost')] = 'GainBoost'
func_list_new[func_list_new.index('accia')] = 'ACCIA'
optimizer_list_temp = [optimizer_list[i][1:] for i in opt_map]
optimizer_list_new = []
for opt_name in optimizer_list_temp:
    if opt_name == "mine":
        opt_name = "MARIO-1"
    if opt_name == "mine_nu":
        opt_name = "MARIO-1se"
    if opt_name == "turbo":
        opt_name = "TuRBO"
    if opt_name == "pycma":
        opt_name = "CMA-ES"
    if opt_name == "pyVTS":
        opt_name = "cVTS"
    if opt_name == "pybobyqa":
        opt_name = "BOBYQA"
    optimizer_list_new.append(opt_name)
#optimizer_list_new = ["mine" if i.startswith("pybobyqaPlus") else i for i in optimizer_list_new]

sum_Z = np.mean(Z, axis=1)
for i, opt_name in enumerate(optimizer_list_new):
    opt_name += "(" + "{:.3f}".format(sum_Z[i]) + ")"
    optimizer_list_new[i] = opt_name

# plot
fig, ax = plt.subplots(figsize=(18,12), layout='constrained')

pc = ax.imshow(Z, origin='lower', vmin=0, vmax=1, cmap='viridis_r') 
ax.set_aspect('auto')
ax.set_xticks(np.arange(len(func_list)), labels=func_list_new, size=26, weight='bold', rotation=70)
#ax.set_xticks([])
#ax.set_xlabel('test functions')
ax.set_yticks(np.arange(len(optimizer_list)), labels=optimizer_list_new, size=30, weight='bold')
ax.get_yticklabels()[-1].set_color("red")
fig.colorbar(pc, ax=ax)

plt.savefig('hh_linear.pdf')
