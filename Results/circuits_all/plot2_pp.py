import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import datetime

#plt.style.use('_mpl-gallery-nogrid')

def transform_perform1(p):
    if np.isinf(p):
        return np.inf
    else:
        assert p >= 0
        assert p <= 1
        tau = 0.5
        if 1-p < tau:
            return (p-0.5)/0.5
        else:
            return 0.

def transform_perform2(p):
    if np.isinf(p):
        return np.inf
    else:
        tau = 0.05
        if 1-p < tau:
            return 1.
        else:
            return 0.

def transform_perform(p):
    if np.isinf(p):
        return np.inf
    return p

def get_bestline(loc, fb):
    loc_files = os.listdir(loc)
    print("at location", loc, "...")
    # print("loc_files {}".format(loc_files))
    data_files = [i for i in loc_files if i.startswith("database") and i.endswith(".pkl")]
    merge_database = []
    for data_file in data_files:
        with open(os.path.join(loc, data_file),"rb") as handler:
            database = pickle.load(handler)
        merge_database.extend(database)
    best_line = []
    i_bound = max([record['number'] for record in merge_database])
    for i in range(fb):
        if i <= i_bound:
            values = [record['value'] for record in merge_database if record['number'] == i]
            best_line.append(min(values))
        else:
            best_line.append(best_line[-1])
    return np.minimum.accumulate(best_line)


# make data
print("start time:", datetime.datetime.now())
csv_file = 'concat.csv'
npy_file1 = 'post_processing/concat.npy'
npy_file2 = 'post_processing_time/concat.npy'

fb_factor = 20
repetitions = 10
vectorized_transform = np.vectorize(transform_perform)

df = pd.read_csv(csv_file)
#df['budget_factor'] = 10
#df['dimension'] = 10
optimizer_list = df['optimizer_name'].unique()
func_list = df['func_name'].unique()
print("optimizers:", optimizer_list)
print("func_names:", func_list)
print(df[df['optimizer_name'] == 'mturbo'])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18,26), layout="constrained")

# evaluation
func_results_matrix = np.load(npy_file1)
print("func_results_matrix", func_results_matrix.shape)
#del_list = []
#for index, func_name in enumerate(func_list):
#    if func_name.startswith("MySinusoidal"):
#        del_list.append(index)
#func_results_matrix = np.delete(func_results_matrix, del_list, axis=0)
#print("func_results_matrix", func_results_matrix.shape)
# axis 0 for funcs, axis 1 for optimizers, axis 2 for repetitions, axis 3 for fb_points + 1
temp_perform_matrix = vectorized_transform(func_results_matrix)
print("temp_perform_matrix", temp_perform_matrix.shape)
#temp_perform_matrix = np.min(vectorized_transform(func_results_matrix), axis=2)
temp_fb = np.min(np.sum(~np.isinf(temp_perform_matrix), axis=-1))
factor_performs = []
for i in range(temp_perform_matrix.shape[0]):
    data = temp_perform_matrix[i]
    data_fb = np.sum(~np.isinf(data), axis=-1)[0][0]
    factor_index = np.clip(np.linspace(0, data_fb-1, temp_fb), 0, data_fb-1).astype(int)
    factor_data = data[:,:,factor_index]
    #print("factor_data", factor_data.shape)
    factor_performs.append(factor_data)
factor_performs = np.array(factor_performs)
print("factor_performs", factor_performs.shape)
avg_perform_matrix = np.mean(factor_performs, axis=(0,2))
std_perform_matrix = np.std(factor_performs, axis=(0,2))
avgm_perform_matrix = np.tan(avg_perform_matrix*np.pi/2)
avgp_perform_matrix = np.tan((avg_perform_matrix+0.05*std_perform_matrix)*np.pi/2)
avgn_perform_matrix = np.tan((avg_perform_matrix-0.05*std_perform_matrix)*np.pi/2)
print("avg_perform_matrix", avg_perform_matrix.shape)

mine_line = None
mine_color = None
#for index in np.argsort(-avg_perform_matrix[:,-1]):
for sort_name, fill_color in zip(
    ["mmine", "mmine_nu", "mturbo", "mpycma", "mDE", "mTBPSA", "mShiwa", "mpyVTS", "mpybobyqa", "mABBO"],
    ["C3", "C0", "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]):
    mine_flag = False
    hline_flag = False
    index = optimizer_list.tolist().index(sort_name)
    opt_name = optimizer_list[index][1:]
    #opt_name = optimizer_list[index][1:]
    if opt_name == "mine":
        opt_name = "MARIO-1"
        mine_flag = True
    if opt_name == "mine_nu":
        opt_name = "MARIO-1se"
        #hline_flag = True
    if opt_name == "turbo":
        opt_name = "TuRBO"
        hline_flag = True
    if opt_name == "pycma":
        opt_name = "CMA-ES"
        #hline_flag = True
    if opt_name == "pyVTS":
        opt_name = "cVTS"
    if opt_name == "pybobyqa":
        opt_name = "BOBYQA"
    data = avgm_perform_matrix[index]
    p_data = avgp_perform_matrix[index]
    n_data = avgn_perform_matrix[index]
    assert len(data) == temp_fb
    x = np.linspace(0,fb_factor,temp_fb+1)
    y = np.hstack([data[0:1], data[:temp_fb]])
    yp = np.hstack([p_data[0:1], p_data[:temp_fb]])
    yn = np.hstack([n_data[0:1], n_data[:temp_fb]])
    if mine_flag:
        p = ax1.plot(x, y, '-', linewidth=5, label=opt_name, color=fill_color)
    else:
        p = ax1.plot(x, y, '-', linewidth=3, label=opt_name, color=fill_color)
    ax1.fill_between(x, yn, yp, alpha=0.2, color=fill_color)
    if mine_flag:
        mine_line = y
        mine_color = fill_color
    if hline_flag:
        hline_val = y[-1]
        cross_x = (np.argmin(np.abs(mine_line-hline_val))/temp_fb)*fb_factor
        print("optimizer", opt_name)
        print("hline_val", hline_val)
        print("cross_x", cross_x)
        ax1.axhline(hline_val, 5/fb_factor, 1, color=mine_color, linestyle='--', linewidth=7.5)
        ax1.text(5, hline_val+0.25, "{:.2f}X".format(fb_factor/cross_x), fontsize=44, fontweight='bold', color=mine_color)

# time
func_results_matrix = np.load(npy_file2)
print("func_results_matrix", func_results_matrix.shape)
#del_list = []
#for index, func_name in enumerate(func_list):
#    if func_name.startswith("MySinusoidal"):
#        del_list.append(index)
#func_results_matrix = np.delete(func_results_matrix, del_list, axis=0)
#print("func_results_matrix", func_results_matrix.shape)
# axis 0 for funcs, axis 1 for optimizers, axis 2 for repetitions, axis 3 for fb_points + 1
temp_perform_matrix = vectorized_transform(func_results_matrix)
print("temp_perform_matrix", temp_perform_matrix.shape)
#temp_perform_matrix = np.min(vectorized_transform(func_results_matrix), axis=2)
temp_fb = np.min(np.sum(~np.isinf(temp_perform_matrix), axis=-1))
factor_performs = []
for i in range(temp_perform_matrix.shape[0]):
    data = temp_perform_matrix[i]
    data_fb = np.sum(~np.isinf(data), axis=-1)[0][0]
    factor_index = np.clip(np.linspace(0, data_fb-1, temp_fb), 0, data_fb-1).astype(int)
    factor_data = data[:,:,factor_index]
    #print("factor_data", factor_data.shape)
    factor_performs.append(factor_data)
factor_performs = np.array(factor_performs)
print("factor_performs", factor_performs.shape)
avg_perform_matrix = np.mean(factor_performs, axis=(0,2))
std_perform_matrix = np.std(factor_performs, axis=(0,2))
avgm_perform_matrix = np.tan(avg_perform_matrix*np.pi/2)
avgp_perform_matrix = np.tan((avg_perform_matrix+0.05*std_perform_matrix)*np.pi/2)
avgn_perform_matrix = np.tan((avg_perform_matrix-0.05*std_perform_matrix)*np.pi/2)
print("avg_perform_matrix", avg_perform_matrix.shape)

mine_line = None
mine_color = None
#for index in np.argsort(-avg_perform_matrix[:,-1]):
for sort_name, fill_color in zip(
    ["mmine", "mmine_nu", "mturbo", "mpycma", "mDE", "mTBPSA", "mShiwa", "mpyVTS", "mpybobyqa", "mABBO"],
    ["C3", "C0", "C1", "C2", "C4", "C5", "C6", "C7", "C8", "C9"]):
    mine_flag = False
    hline_flag = False
    index = optimizer_list.tolist().index(sort_name)
    opt_name = optimizer_list[index][1:]
    #opt_name = optimizer_list[index][1:]
    if opt_name == "mine":
        opt_name = "MARIO-1"
        mine_flag = True
    if opt_name == "mine_nu":
        opt_name = "MARIO-1se"
        #hline_flag = True
    if opt_name == "turbo":
        opt_name = "TuRBO"
        hline_flag = True
    if opt_name == "pycma":
        opt_name = "CMA-ES"
        #hline_flag = True
    if opt_name == "pyVTS":
        opt_name = "cVTS"
    if opt_name == "pybobyqa":
        opt_name = "BOBYQA"
    data = avgm_perform_matrix[index]
    p_data = avgp_perform_matrix[index]
    n_data = avgn_perform_matrix[index]
    assert len(data) == temp_fb
    x = np.linspace(0,fb_factor,temp_fb+1)
    y = np.hstack([data[0:1], data[:temp_fb]])
    yp = np.hstack([p_data[0:1], p_data[:temp_fb]])
    yn = np.hstack([n_data[0:1], n_data[:temp_fb]])
    if mine_flag:
        p = ax2.plot(x, y, '-', linewidth=5, label=opt_name, color=fill_color)
    else:
        p = ax2.plot(x, y, '-', linewidth=3, label=opt_name, color=fill_color)
    ax2.fill_between(x, yn, yp, alpha=0.2, color=fill_color)
    if mine_flag:
        mine_line = y
        mine_color = fill_color
    if hline_flag:
        hline_val = y[-1]
        cross_x = (np.argmin(np.abs(mine_line-hline_val))/temp_fb)*fb_factor
        print("optimizer", opt_name)
        print("hline_val", hline_val)
        print("cross_x", cross_x)
        ax2.axhline(hline_val, 5/fb_factor, 1, color=mine_color, linestyle='--', linewidth=7.5)
        ax2.text(5, hline_val+0.25, "{:.2f}X".format(fb_factor/cross_x), fontsize=44, fontweight='bold', color=mine_color)

ax1.set_xlim(0, fb_factor)
ax2.set_xlim(0, fb_factor)
ax1.set_xlabel('Normalized number of evaluations', fontdict={'size':32, 'weight':'bold'})
ax2.set_xlabel('Normalized optimization time', fontdict={'size':32, 'weight':'bold'})
ax1.set_ylim(bottom=0, top=3)
ax2.set_ylim(bottom=0, top=3)
ax1.set_ylabel('Tangent-transformed data profiles', fontdict={'size':32, 'weight':'bold'})
ax2.set_ylabel('Tangent-transformed data profiles', fontdict={'size':32, 'weight':'bold'})
ax1.grid()
ax2.grid()
#ax.legend()
ax_handles, ax_labels = ax1.get_legend_handles_labels()
fig.legend(handles=ax_handles, labels=ax_labels, ncol=5, bbox_to_anchor=(0.5,0), loc='upper center', prop = {'size':27, 'weight':'bold'})
# lines, labels = fig.axes[-1].get_legend_handles_labels()
# fig.legend(handles=lines, labels=labels, ncol=2, bbox_to_anchor=(0.5, 0), loc='upper center', prop = {'size':38, 'weight':'bold'})
plt.xticks(size=15)
plt.yticks(size=15)
plt.savefig("pp2_tan.pdf", bbox_inches='tight')
print("end time:", datetime.datetime.now())
