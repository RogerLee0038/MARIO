import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import math

#print(matplotlib.get_backend())
plt.figure()
lines=[] #data of curves to be ploted. item:line line: ponts of a curve
names=[] #name of curves
for index, algo in enumerate(sys.argv[1:]): #for different optimization results
    databaseFile = "database_{}.pkl".format(algo)
    #invalid cases
    if not os.path.exists(databaseFile):
        print("There is no "+databaseFile+"!")
        sys.exit()
    else: #normal case, use splite3 to get the data of cost vs simulation_num, store in plot_list
        with open(databaseFile, "rb") as handler:
            database = pickle.load(handler)
        plot_list = [record['value'] for record in database]

        #find by which simulation_num the cost is minimized
        mini=min(plot_list)
        best=round(mini,4)
        best_index=plot_list.index(mini)

        #change the cost vs simulation_num to best_cost vs simulation_num
        plot_array = np.array(plot_list)
        plot_list = np.minimum.accumulate(plot_array).tolist()
        #print(plot_list)
        
        print(databaseFile + " total_simulation_num = " + str(len(plot_list)))

        #plot one optimization curve with annotated first minimum information
        x=np.array(range(len(plot_list)))
        y=np.array(plot_list)
        line, =plt.plot(x+1,y)
        lines.append(line)
        names.append(algo)
        #print(lines)
        #print(names)
        plt.scatter(best_index+1,best)
        plt.annotate("("+str(best_index+1)+","+str(best)+")", xy = (best_index+1, best), xytext = (best_index+1.5, best),
                    arrowprops = {
                        'headwidth': 10, # 箭头头部的宽度
                        'headlength': 5, # 箭头头部的长度
                        'width': 2, # 箭头尾部的宽度
                        'facecolor': line.get_color(), # 箭头的颜色
                        'shrink': 0.1, # 从箭尾到标注文本内容开始两端空隙长度
                        },
                    #family='Times New Roman',  # 标注文本字体为Times New Roman
                    fontsize=10,  # 文本大小为10
                    #fontweight='bold',  # 文本为粗体
                    color='black',  # 文本颜色
                    # ha = 'center' # 水平居中
           )
           
plt.legend(handles=lines,labels=names,loc="upper right",fontsize=6)
plt.xlabel("simulation num")
plt.ylabel("best cost")
plt.show()
