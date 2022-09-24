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

#color_set = ['#63b2ee','#f8cb7f','#76da91','#f89588']

color_set = ['#f8cb7f','#76da91','#63b2ee']

#below are 1 vs 4 codes
#load model 

#firstly, generate IQL models
#score_1 = np.load('base_4_methods_10_seeds/score_mean.npy')[0:10,:]
#similarity_1 = np.load('base_4_methods_10_seeds/distance_b2a.npy')[0:10,:]
'''
import os
exp_name_1 = 't1r02_sad'
exp_name_2 = 't2r015_sad'
score_split = np.load('t1t2_sad/score_mean.npy')
score_sim = np.load('t1t2_sad/distance_b2a.npy')
os.makedirs(exp_name_1)
os.makedirs(exp_name_2)
np.save(exp_name_1+ "/score_mean.npy", score_split[0:10,:])
np.save(exp_name_1+ "/distance_b2a.npy", score_sim[0:10,:])
np.save(exp_name_2+ "/score_mean.npy", score_split[10:,:])
np.save(exp_name_2+ "/distance_b2a.npy", score_sim[10:,:])
'''



base_score = np.load('base_4_methods_10_seeds/score_mean.npy')
base_sim = np.load('base_4_methods_10_seeds/distance_b2a.npy')

iql_score = base_score[0:10,:].flatten()
iql_sim = base_sim[0:10,:].flatten()

vdn_score = base_score[10:20,:].flatten()
vdn_sim = base_sim[10:20,:].flatten()

sad_score = base_score[20:30,:].flatten()
sad_sim = base_sim[20:30,:].flatten()


name = ['t1r03_iql','t2r015_iql','t3r02_iql','t1r03_vdn','t2r015_vdn','t3r02_vdn','t1r02_sad','t2r015_sad','t3r02_sad']
base_scores = [iql_score,vdn_score,sad_score]
base_sims = [iql_sim,vdn_sim,sad_sim]

iql_a_score = np.load('base_4_methods_10_seeds/score_mean.npy')

model_names = ['IQL','VDN','SAD']
type_names = ['SBRT(A)','SBRT(B)','SBRT(C)']

for model_idx in range(3):
    for type_idx in range(3):
        base_name = model_names[model_idx]
        compared_name = base_name +"+"+type_names[type_idx]
        label_set = [base_name,compared_name]
        model_name = name[model_idx*3+type_idx]
        score_new = np.load(model_name+'/score_mean.npy').flatten()
        sim_new = np.load(model_name+'/distance_b2a.npy').flatten()

        sim_set = [base_sims[model_idx],sim_new]
        score_set = [base_scores[model_idx],score_new]

        fig_title = base_name+' vs '+compared_name
        file_name = model_name

        #plot and analysis
        plt.figure()
        for idx in range(len(label_set)):
            x_data, y_data = sim_set[idx].flatten(), score_set[idx].flatten()
            y_data = np.delete(y_data,x_data>0.95)
            x_data = np.delete(x_data,x_data>0.95)
            plt.scatter(x_data, y_data, c=color_set[idx], alpha=0.6, label=label_set[idx])
            k,b = np.polyfit(x_data,y_data,1)
            x = np.linspace(0,1,50)
            y = k*x + b
            print('label:',label_set[idx],' k:',k,' b:',b, ' person coef:',pearson_coef(x_data, y_data))
            plt.plot(x,y,c=color_set[idx],lw=2)
        plt.title(fig_title,fontsize=22)    

        plt.xlim(0, 1)
        plt.ylim(0, 25)
        plt.ylabel('Cross-Play Scores', fontsize=18)
        plt.xlabel('Conditional Policy Similarity', fontsize=18)
        plt.legend(fontsize=14, loc='lower right')
        plt.savefig('figs/'+file_name+'.pdf', bbox_inches='tight')
'''
score = np.load('nosharing_total/score_mean.npy').flatten()
similarity = np.load('nosharing_total/distance_b2a.npy').flatten()
score = np.delete(score,similarity>0.95)
similarity = np.delete(similarity,similarity>0.95)
print(pearson_coef(similarity, score))

score_1 = np.load('t3r02_iql/score_mean.npy')
similarity_1 = np.load('t3r02_iql/distance_b2a.npy')

score_2 = np.load('t3r04_iql/score_mean.npy')
similarity_2 = np.load('t3r04_iql/distance_b2a.npy')

label_set = [r"$\alpha_r$=0.8",r"$\alpha_r$=0.6"]
score_set = [score_1,score_2]
similarity_set = [similarity_1,similarity_2]
fig_title = 'IQL+SBRT'
file_name = 'iql_alphachoose'

#plot and analysis
plt.figure()
for idx in range(len(label_set)):
    x_data, y_data = similarity_set[idx].flatten(), score_set[idx].flatten()
    y_data = np.delete(y_data,x_data>0.95)
    x_data = np.delete(x_data,x_data>0.95)
    plt.scatter(x_data, y_data, c=color_set[idx], alpha=0.6, label=label_set[idx])
    k,b = np.polyfit(x_data,y_data,1)
    x = np.linspace(0,1,50)
    y = k*x + b
    print('label:',label_set[idx],' k:',k,' b:',b, ' person coef:',pearson_coef(x_data, y_data))
    #plt.plot(x,y,c=color_set[idx],lw=2)
plt.title(fig_title,fontsize=22)    

plt.xlim(0, 1)
plt.ylim(0, 25)
plt.ylabel('Cross-Play Scores', fontsize=18)
plt.xlabel('Conditional Policy Similarity', fontsize=18)
plt.legend(fontsize=14, loc='lower right')
plt.savefig('figs/'+file_name+'.pdf', bbox_inches='tight')


score_1 = np.load('t3r02_vdn/score_mean.npy')
similarity_1 = np.load('t3r02_vdn/distance_b2a.npy')

score_2 = np.load('sad_vdn_r04/score_mean.npy')[10:,:]
similarity_2 = np.load('sad_vdn_r04/distance_b2a.npy')[10:,:]

label_set = [r"$\alpha_r$=0.8",r"$\alpha_r$=0.6"]
score_set = [score_1,score_2]
similarity_set = [similarity_1,similarity_2]
fig_title = 'VDN+SBRT'
file_name = 'vdn_alphachoose'

#plot and analysis
plt.figure()
for idx in range(len(label_set)):
    x_data, y_data = similarity_set[idx].flatten(), score_set[idx].flatten()
    y_data = np.delete(y_data,x_data>0.95)
    x_data = np.delete(x_data,x_data>0.95)
    plt.scatter(x_data, y_data, c=color_set[idx], alpha=0.6, label=label_set[idx])
    k,b = np.polyfit(x_data,y_data,1)
    x = np.linspace(0,1,50)
    y = k*x + b
    print('label:',label_set[idx],' k:',k,' b:',b, ' person coef:',pearson_coef(x_data, y_data))
    #plt.plot(x,y,c=color_set[idx],lw=2)
plt.title(fig_title,fontsize=22)    

plt.xlim(0, 1)
plt.ylim(0, 25)
plt.ylabel('Cross-Play Scores', fontsize=18)
plt.xlabel('Conditional Policy Similarity', fontsize=18)
plt.legend(fontsize=14, loc='lower right')
plt.savefig('figs/'+file_name+'.pdf', bbox_inches='tight')




score_1 = np.load('t3r02_sad/score_mean.npy')
similarity_1 = np.load('t3r02_sad/distance_b2a.npy')

score_2 = np.load('sad_vdn_r04/score_mean.npy')[0:10,:]
similarity_2 = np.load('sad_vdn_r04/distance_b2a.npy')[0:10,:]

label_set = [r"$\alpha_r$=0.8",r"$\alpha_r$=0.6"]
score_set = [score_1,score_2]
similarity_set = [similarity_1,similarity_2]
fig_title = 'SAD+SBRT'
file_name = 'sad_alphachoose'

#plot and analysis
plt.figure()
for idx in range(len(label_set)):
    x_data, y_data = similarity_set[idx].flatten(), score_set[idx].flatten()
    y_data = np.delete(y_data,x_data>0.95)
    x_data = np.delete(x_data,x_data>0.95)
    plt.scatter(x_data, y_data, c=color_set[idx], alpha=0.6, label=label_set[idx])
    k,b = np.polyfit(x_data,y_data,1)
    x = np.linspace(0,1,50)
    y = k*x + b
    print('label:',label_set[idx],' k:',k,' b:',b, ' person coef:',pearson_coef(x_data, y_data))
    #plt.plot(x,y,c=color_set[idx],lw=2)
plt.title(fig_title,fontsize=22)    

plt.xlim(0, 1)
plt.ylim(0, 25)
plt.ylabel('Cross-Play Scores', fontsize=18)
plt.xlabel('Conditional Policy Similarity', fontsize=18)
plt.legend(fontsize=14, loc='lower right')
plt.savefig('figs/'+file_name+'.pdf', bbox_inches='tight')

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

