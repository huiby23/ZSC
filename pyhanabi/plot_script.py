# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def pearson_coef(x,y):
    X = np.vstack([x,y])
    return np.corrcoef(X)[0][1]

score_mean = np.load('D:/Git/ZSC/base_4_methods_5_seeds/score_mean.npy')
distance_a2b = np.load('D:/Git/ZSC/base_4_methods_5_seeds/distance_a2b.npy')

for idx in range(20):
    score_mean[idx,:] = score_mean[idx,:]/score_mean[idx,idx]

name_set = ['vdn','sad','op','iql']
color_set = ['#0082fc','#fdd845','#22ed7c','#f47a75']

coef_mat = np.zeros((4,4))

#Task 1: plot 4x4  
for a_idx in range(4):
    plt.figure()
    for b_idx in range(4):
        y_data = score_mean[a_idx*5:a_idx*5+5,b_idx*5:b_idx*5+5].flatten()
        x_data = distance_a2b[a_idx*5:a_idx*5+5,b_idx*5:b_idx*5+5].flatten()
        coef_mat[a_idx,b_idx] = pearson_coef(x_data, y_data)

        plt.scatter(x_data, y_data, c=color_set[b_idx], alpha=1,label=name_set[b_idx])
    title = name_set[a_idx]+' with others'
    plt.title(title,fontsize=22)    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel('performance', fontsize=18)
    plt.xlabel('similarity', fontsize=18)
    plt.legend(fontsize=14, loc='lower right')
    plt.savefig('D:/Git/ZSC/figures/model_'+name_set[a_idx]+'.pdf', bbox_inches='tight')

