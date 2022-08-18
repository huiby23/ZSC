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
'''
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
'''
#Task 1: plot 4x4  

y_data = np.array([ 0.367,  0.24 ,  0.048,  4.497,  4.467, 14.854, 14.554, 12.292, 14.696, 15.132, 13.995, 13.438, 12.75 , 14.396, 12.649,  6.871, 10.643,  9.834,  0.961,  9.303])/23.5
x_data = np.array([0.27113099, 0.18783245, 0.18722341, 0.28541259, 0.33509939, 0.39627468, 0.38555464, 0.44437809, 0.39790028, 0.40730822, 0.39609578, 0.38074237, 0.39335643, 0.45994792, 0.39262473, 0.38697767, 0.34996601, 0.36162021, 0.37420798, 0.35580595])

plt.scatter(x_data, y_data, c=color_set[0], alpha=0.7,label='ours')
title = 'ours with others'
plt.title(title,fontsize=22)    
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.ylabel('performance', fontsize=18)
plt.xlabel('similarity', fontsize=18)
plt.legend(fontsize=14, loc='lower right')
plt.show()
#plt.savefig('D:/Git/ZSC/figures/model_'+name_set[a_idx]+'.pdf', bbox_inches='tight')
