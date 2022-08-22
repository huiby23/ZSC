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
'''
score_mean = np.load('D:/Git/ZSC/base_4_methods_5_seeds/score_mean.npy')
distance_a2b = np.load('D:/Git/ZSC/base_4_methods_5_seeds/distance_a2b.npy')

for idx in range(20):
    score_mean[idx,:] = score_mean[idx,:]/score_mean[idx,idx]

name_set = ['vdn','sad','op','iql']
color_set = ['#0082fc','#fdd845','#22ed7c','#f47a75']

coef_mat = np.zeros((4,4))

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

y_data = np.array([ 7.157,  3.443,  0.926,  9.5,   12.181, 22.293, 21.986, 21.26,  21.483, 21.603, 22.285, 21.569, 22.224, 21.728, 16.281, 19.978, 18.943, 15.664, 19.823])
x_data = np.array([0.33555618, 0.21537602, 0.19380638, 0.49208951, 0.58804387, 0.6551071, 0.64811841, 0.61541136, 0.6290753,  0.65994793, 0.69532983, 0.64431419, 0.70458598, 0.66127302, 0.46650904, 0.67208433, 0.67823531, 0.49094989, 0.67777062])
'''
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
'''
k,b = np.polyfit(x_data,y_data,1)
p_coef = pearson_coef(x_data, y_data)
point_1,point_2,point_3 = 0.4*k+b, 0.6*k+b, 0.8*k+b
print('k:',"%.2f" %k)
print('b:',"%.2f" %b)
print('pearson coefficient:', "%.2f" %p_coef)
print('0.4:',"%.2f"%point_1)
print('0.6:',"%.2f"%point_2)
print('0.8:',"%.2f"%point_3)
