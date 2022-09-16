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

color_set = ['#63b2ee','#f8cb7f','#76da91','#f89588']





#below are 1 vs 4 codes
#load model 

score_1 = np.load('base_4_methods_10_seeds/score_mean.npy')[10:20,:]
similarity_1 = np.load('base_4_methods_10_seeds/distance_b2a.npy')[10:20,:]

score_2 = np.load('t1r03_vdn/score_mean.npy')
similarity_2 = np.load('t1r03_vdn/distance_b2a.npy')

label_set = ['VDN','VDN+SBT(A)']
score_set = [score_1,score_2]
similarity_set = [similarity_1,similarity_2]
fig_title = 'VDN vs VDN+SBT(A)'
file_name = 'vdn_vs_sbt_a'

#plot and analysis
plt.figure()
for idx in range(len(label_set)):
    x_data, y_data = similarity_set[idx].flatten(), score_set[idx].flatten()
    y_data = np.delete(y_data,x_data>0.95)
    x_data = np.delete(x_data,x_data>0.95)
    plt.scatter(x_data, y_data, c=color_set[idx], alpha=0.7, label=label_set[idx])
    k,b = np.polyfit(x_data,y_data,1)
    x = np.linspace(0,1,50)
    y = k*x + b
    print('label:',label_set[idx],' k:',k,' b:',b, ' person coef:',pearson_coef(x_data, y_data))
    plt.plot(x,y,c=color_set[idx])
plt.title(fig_title,fontsize=22)    



plt.xlim(0, 1)
plt.ylim(0, 25)
plt.ylabel('performance', fontsize=18)
plt.xlabel('similarity', fontsize=18)
plt.legend(fontsize=14, loc='lower right')
plt.savefig('figs/'+file_name+'.pdf', bbox_inches='tight')

'''


#below are 4 vs 4 codes

score_mean = np.load('base_4_methods_10_seeds/score_mean.npy')
distance_b2a = np.load('base_4_methods_10_seeds/distance_b2a.npy')

name_set = ['IQL','VDN','SAD','OBL']
color_set = ['#63b2ee','#f8cb7f','#76da91','#f89588']
fig_title = ['IQL vs Others','VDN vs Others','SAD vs Others','OBL vs Others']
file_name = 'basetest'

coef_mat = np.zeros((4,4))

for a_idx in range(4):
    plt.figure()
    for b_idx in range(4):
        y_data = score_mean[a_idx*10:a_idx*10+10,b_idx*10:b_idx*10+10].flatten()
        x_data = distance_b2a[a_idx*10:a_idx*10+10,b_idx*10:b_idx*10+10].flatten()
        y_data = np.delete(y_data,x_data>0.95)
        x_data = np.delete(x_data,x_data>0.95)
        coef_mat[a_idx,b_idx] = pearson_coef(x_data, y_data)

        plt.scatter(x_data, y_data, c=color_set[b_idx], alpha=0.7,label=name_set[b_idx])
    y_data = score_mean[a_idx*10:a_idx*10+10,:].flatten()
    x_data = distance_b2a[a_idx*10:a_idx*10+10,:].flatten()   
    y_data = np.delete(y_data,x_data>0.95)
    x_data = np.delete(x_data,x_data>0.95)
    k,b = np.polyfit(x_data,y_data,1)
    x = np.linspace(0,1,50)
    y = k*x + b
    print('model:',name_set[a_idx],' k:',k,' b:',b, ' person coef:',pearson_coef(x_data, y_data))
    plt.plot(x,y,c='black') 
    plt.title(fig_title[a_idx],fontsize=22)    
    plt.xlim(0, 1)
    plt.ylim(0, 25)
    plt.ylabel('performance', fontsize=18)
    plt.xlabel('similarity', fontsize=18)
    plt.legend(fontsize=14, loc='lower right')
    plt.savefig('figs/'+file_name+str(a_idx)+'.pdf', bbox_inches='tight')

'''

