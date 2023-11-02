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

color_set = ['#f8cb7f','#76da91','#63b2ee','#f89588']


#below are 4 vs 4 codes
#————————————————————————————————————————————————————————————————————————————————————————————#
score_mean = np.load('base_4_methods_10_seeds/score_mean.npy')[0:30,0:30]
distance_b2a = np.load('base_4_methods_10_seeds/distance_b2a.npy')[0:30,0:30]
score_new = np.load('ssd_models_cp/score_mean.npy')
distance_new_a2b = np.load('ssd_models_cp/distance_a2b.npy')
distance_new_b2a = np.load('ssd_models_cp/distance_b2a.npy')
score_intra = np.load('ssd_intra/score_mean.npy')
distance_intra = np.load('ssd_intra/distance_b2a.npy')

name_set = ['IQL','VDN','AUX','SAD']
color_set = ['#63b2ee','#f8cb7f','#76da91','#f89588']
fig_title = ['IQL vs All','VDN vs All','AUX vs All','SAD vs ALL']
file_name = 'four_basetest'

cross_play_score_mean = np.zeros((4,4))
cross_play_score_ste = np.zeros((4,4))

for a_idx in range(3):
    plt.figure()
    for b_idx in range(3):
        y_data = score_mean[a_idx*10:a_idx*10+10,b_idx*10:b_idx*10+10].flatten()
        x_data = distance_b2a[a_idx*10:a_idx*10+10,b_idx*10:b_idx*10+10].flatten()
        y_data = np.delete(y_data,x_data>0.95)
        x_data = np.delete(x_data,x_data>0.95)
        plt.scatter(x_data, y_data, c=color_set[b_idx], alpha=0.7,label=name_set[b_idx])

        cross_play_score_mean[a_idx,b_idx] = np.mean(y_data)
        
        if a_idx!=b_idx:
            cross_play_score_ste[a_idx,b_idx] = np.std(y_data)/10
        else:
            cross_play_score_ste[a_idx,b_idx] = np.std(y_data)/7

    y_data = score_new[0:10,a_idx*30:a_idx*30+10].flatten()
    x_data = distance_new_a2b[0:10,a_idx*30:a_idx*30+10].flatten()    
    plt.scatter(x_data, y_data, c=color_set[3], alpha=0.7,label=name_set[3])

    cross_play_score_mean[a_idx,3] = np.mean(y_data)
    cross_play_score_ste[a_idx,3] = np.std(y_data)/10


    y_data_base = score_mean[a_idx*10:a_idx*10+10,:]
    x_data_base = distance_b2a[a_idx*10:a_idx*10+10,:]  

    y_data_new = score_new[0:10,a_idx*30:a_idx*30+10]
    x_data_new = distance_new_a2b[0:10,a_idx*30:a_idx*30+10]

    y_data = np.hstack((y_data_base,y_data_new)).flatten()
    x_data = np.hstack((x_data_base,x_data_new)).flatten()

    y_data = np.delete(y_data,x_data>0.95)
    x_data = np.delete(x_data,x_data>0.95)

    k,b = np.polyfit(x_data,y_data,1)
    x = np.linspace(0,1,50)
    y = k*x + b
    print('model:',name_set[a_idx],' k:',k,' b:',b, ' person coef:',pearson_coef(x_data, y_data))
    #plt.plot(x,y,c='black') 
    plt.title(fig_title[a_idx],fontsize=28)    
    plt.xlim(0, 1)
    plt.ylim(0, 25)
    plt.ylabel('Cross-Play Scores', fontsize=24)
    plt.xlabel('CPSTT', fontsize=24)
    plt.legend(fontsize=20, loc='lower right')
    plt.savefig('figs/'+file_name+str(a_idx)+'.pdf', bbox_inches='tight')

#now it's time for SSD
plt.figure()
for b_idx in range(3):
    y_data = score_new[0:10,b_idx*30:b_idx*30+10].flatten()
    x_data = distance_new_b2a[0:10,b_idx*30:b_idx*30+10].flatten()   

    plt.scatter(x_data, y_data, c=color_set[b_idx], alpha=0.7,label=name_set[b_idx])

y_data = score_intra.flatten()
x_data = distance_intra.flatten()   
y_data = np.delete(y_data,x_data>0.95)
x_data = np.delete(x_data,x_data>0.95) 
plt.scatter(x_data, y_data, c=color_set[3], alpha=0.7,label=name_set[3])
print('intra-play of SAD mean:', np.mean(y_data))
print('intra-play of SAD ste:', np.std(y_data)/7)

y_data = np.hstack((score_new[0:10,0:10],score_new[0:10,30:40],score_new[0:10,60:70],score_intra)).flatten()
x_data = np.hstack((distance_new_b2a[0:10,0:10],distance_new_b2a[0:10,30:40],distance_new_b2a[0:10,60:70],distance_intra)).flatten()
y_data = np.delete(y_data,x_data>0.95)
x_data = np.delete(x_data,x_data>0.95) 
k,b = np.polyfit(x_data,y_data,1)
x = np.linspace(0,1,50)
y = k*x + b
print('model:',name_set[3],' k:',k,' b:',b, ' person coef:',pearson_coef(x_data, y_data))
#plt.plot(x,y,c='black') 
plt.title(fig_title[3],fontsize=28)    
plt.xlim(0, 1)
plt.ylim(0, 25)
plt.ylabel('Cross-Play Scores', fontsize=24)
plt.xlabel('CPSTT', fontsize=24)
plt.legend(fontsize=20, loc='lower right')
plt.savefig('figs/'+file_name+'3.pdf', bbox_inches='tight')

print('cross-play-mean:', cross_play_score_mean)
print('cross-play-ste:', cross_play_score_ste)




#————————————————————————————————————————————————————————————————————————————————————————————#
#below are OP vs SBRT codes
color_set = ['#f8cb7f','#76da91','#63b2ee','#f89588']

sbrt_name = ['t3r02_iql','t3r02_vdn','t3r02_sad']
op_name = ['iql_op','vdn_op','sad_op']
model_type = ['IQL','VDN','AUX','SAD']
base_score_mean = np.load('base_4_methods_10_seeds/score_mean.npy')[0:30,0:30]
base_distance_b2a = np.load('base_4_methods_10_seeds/distance_b2a.npy')[0:30,0:30]

op_intra_name = ['iql_op_selfplay','vdn_op_intraplay','sad_op_selfplay','ssdop_intra']
sbrt_intra_name = ['t3r02_iql_interplay','t3r02_vdn_interplay','t3r02_sad_interplay','ssdsbrt_intra']

#firstly,add the base three with oldcp results and newcp results
#secondly, deal with new results

op_mean_set = [np.load('iql_op/score_mean.npy')[:,0:30], np.load('vdn_op/score_mean.npy')[:,0:30], np.load('sad_op/score_mean.npy')[:,0:30]]
op_b2a_set = [np.load('iql_op/distance_b2a.npy')[:,0:30], np.load('vdn_op/distance_b2a.npy')[:,0:30], np.load('sad_op/distance_b2a.npy')[:,0:30]]
op_a2b_set = [np.load('iql_op/distance_a2b.npy')[:,0:30], np.load('vdn_op/distance_a2b.npy')[:,0:30], np.load('sad_op/distance_a2b.npy')[:,0:30]]

sbrt_mean_set = [np.load('t3r02_iql/score_mean.npy')[:,0:30], np.load('t3r02_vdn/score_mean.npy')[:,0:30], np.load('t3r02_sad/score_mean.npy')[:,0:30]]
sbrt_b2a_set = [np.load('t3r02_iql/distance_b2a.npy')[:,0:30], np.load('t3r02_vdn/distance_b2a.npy')[:,0:30], np.load('t3r02_sad/distance_b2a.npy')[:,0:30]]
sbrt_a2b_set = [np.load('t3r02_iql/distance_a2b.npy')[:,0:30], np.load('t3r02_vdn/distance_a2b.npy')[:,0:30], np.load('t3r02_sad/distance_a2b.npy')[:,0:30]]

old_cp_mean = np.load('old_models_cp/score_mean.npy')
old_cp_b2a = np.load('old_models_cp/distance_b2a.npy')

new_models_cp_mean = np.load('ssd_models_cp/score_mean.npy')
new_models_cp_b2a = np.load('ssd_models_cp/distance_b2a.npy')
new_models_cp_a2b = np.load('ssd_models_cp/distance_a2b.npy')






#对于新模型而言，直接拿新矩阵里的数据就行了，简单的很

for model_idx in range(3):
    plt.figure()
    #1 base model
    #对于3基础模型而言，首先是基础矩阵里刨除掉self-play的（或者说可以把这个拿出来），然后是在各个op set和sbrt set中拿自己位置对应的mean和a2b，然后是去新矩阵拿列对应的mean和a2b
    #1-1 base vs base
    base_cp_mean = [base_score_mean[model_idx*10:(model_idx+1)*10, 0:10],base_score_mean[model_idx*10:(model_idx+1)*10, 10:20],base_score_mean[model_idx*10:(model_idx+1)*10, 20:30]]
    base_cp_sim = [base_distance_b2a[model_idx*10:(model_idx+1)*10, 0:10],base_distance_b2a[model_idx*10:(model_idx+1)*10, 10:20],base_distance_b2a[model_idx*10:(model_idx+1)*10, 20:30]]
    intra_mean = base_cp_mean.pop(model_idx)
    intra_sim = base_cp_sim.pop(model_idx)
    oldcp_mean = []
    oldcp_sim = []
    #1-2 base vs opsbrt
    for inner_idx in range(3):
        if inner_idx!=model_idx:
            oldcp_mean.append(op_mean_set[inner_idx][:,model_idx*10:(model_idx+1)*10])
            oldcp_sim.append(op_a2b_set[inner_idx][:,model_idx*10:(model_idx+1)*10])
            oldcp_mean.append(sbrt_mean_set[inner_idx][:,model_idx*10:(model_idx+1)*10])
            oldcp_sim.append(sbrt_a2b_set[inner_idx][:,model_idx*10:(model_idx+1)*10])
    #1-3 base vs new
    newmodel_mean = new_models_cp_mean[:,model_idx*30:model_idx*30+10].T
    newmodel_sim = new_models_cp_a2b[:,model_idx*30:model_idx*30+10].T
    #now get results!
    total_cp_mean = base_cp_mean + oldcp_mean
    total_cp_sim = base_cp_sim + oldcp_sim
    total_cp_mean.append(newmodel_mean)
    total_cp_sim.append(newmodel_sim)
    total_cp_mean = np.hstack(total_cp_mean)
    total_cp_sim = np.hstack(total_cp_sim)
    #output scores!
    print(model_type[model_idx],'-base, intercp mean:',np.mean(total_cp_mean),np.std(total_cp_mean)/30)
    #generate datapoints
    y_data = np.concatenate((total_cp_mean, intra_mean),axis=1).flatten()
    x_data = np.concatenate((total_cp_sim, intra_sim),axis=1).flatten()
    y_data = np.delete(y_data,x_data>0.95)
    x_data = np.delete(x_data,x_data>0.95)
    plt.scatter(x_data, y_data, c='#f8cb7f', alpha=0.6, label=model_type[model_idx])
    k,b = np.polyfit(x_data,y_data,1)
    x = np.linspace(0,1,50)
    y = k*x + b
    plt.plot(x,y,c='#f8cb7f',lw=2)
    #2 op model
    #对于3op模型而言，首先是拿自己对应set里刨除掉自己类型算法的mean和b2a，然后是去oldcp里面拿mean和b2a，最后是去新矩阵拿列对应的mean和a2b
    delete_idx = range(10*model_idx,10*(model_idx+1))
    base_cp_mean = np.delete(op_mean_set[model_idx], delete_idx, axis = 1) 
    base_cp_sim = np.delete(op_b2a_set[model_idx], delete_idx, axis = 1)
    #2-2
    delete_idx = range(20*model_idx,20*(model_idx+1))
    three_cp_mean =  np.delete(old_cp_mean[20*model_idx:20*model_idx+10,:], delete_idx, axis = 1)
    three_cp_sim =  np.delete(old_cp_b2a[20*model_idx:20*model_idx+10,:], delete_idx, axis = 1)
    #3-3
    newmodel_mean = new_models_cp_mean[:,model_idx*30+10:model_idx*30+20].T
    newmodel_sim = new_models_cp_a2b[:,model_idx*30+10:model_idx*30+20].T
    #
    total_cp_mean = np.hstack((base_cp_mean, three_cp_mean, newmodel_mean))
    total_cp_sim = np.hstack((base_cp_sim, three_cp_sim, newmodel_sim))
    print(model_type[model_idx],'-op, intercp mean:',np.mean(total_cp_mean),np.std(total_cp_mean)/30)
    #
    intra_mean = np.load(op_intra_name[model_idx]+'/score_mean.npy')
    intra_sim = np.load(op_intra_name[model_idx]+'/distance_b2a.npy')
    #generate datapoints
    y_data = np.concatenate((total_cp_mean, intra_mean),axis=1).flatten()
    x_data = np.concatenate((total_cp_sim, intra_sim),axis=1).flatten()
    y_data = np.delete(y_data,x_data>0.95)
    x_data = np.delete(x_data,x_data>0.95)
    plt.scatter(x_data, y_data, c='#76da91', alpha=0.6, label=model_type[model_idx]+'+OP')
    k,b = np.polyfit(x_data,y_data,1)
    x = np.linspace(0,1,50)
    y = k*x + b
    plt.plot(x,y,c='#76da91',lw=2)
    #3 SBRT model
    #对于3sbrt模型而言，首先是拿自己对应set里刨除掉自己类型算法的mean和b2a，然后是去oldcp里面拿mean和b2a，最后是去新矩阵拿列对应的mean和a2b
    delete_idx = range(10*model_idx,10*(model_idx+1))
    base_cp_mean = np.delete(sbrt_mean_set[model_idx], delete_idx, axis = 1) 
    base_cp_sim = np.delete(sbrt_b2a_set[model_idx], delete_idx, axis = 1)
    #2-2
    delete_idx = range(20*model_idx,20*(model_idx+1))
    three_cp_mean =  np.delete(old_cp_mean[20*model_idx+10:20*model_idx+20,:], delete_idx, axis = 1)
    three_cp_sim =  np.delete(old_cp_b2a[20*model_idx+10:20*model_idx+20,:], delete_idx, axis = 1)
    #3-3
    newmodel_mean = new_models_cp_mean[:,model_idx*30+20:model_idx*30+30].T
    newmodel_sim = new_models_cp_a2b[:,model_idx*30+20:model_idx*30+30].T
    #
    total_cp_mean = np.hstack((base_cp_mean, three_cp_mean, newmodel_mean))
    total_cp_sim = np.hstack((base_cp_sim, three_cp_sim, newmodel_sim))
    print(model_type[model_idx],'-sbrt, intercp mean:',np.mean(total_cp_mean),np.std(total_cp_mean)/30)
    #
    intra_mean = np.load(sbrt_intra_name[model_idx]+'/score_mean.npy')
    intra_sim = np.load(sbrt_intra_name[model_idx]+'/distance_b2a.npy')
    #generate datapoints
    y_data = np.concatenate((total_cp_mean, intra_mean),axis=1).flatten()
    x_data = np.concatenate((total_cp_sim, intra_sim),axis=1).flatten()
    y_data = np.delete(y_data,x_data>0.95)
    x_data = np.delete(x_data,x_data>0.95)
    plt.scatter(x_data, y_data, c='#63b2ee', alpha=0.6, label=model_type[model_idx]+'+SBRT')
    k,b = np.polyfit(x_data,y_data,1)
    x = np.linspace(0,1,50)
    y = k*x + b
    plt.plot(x,y,c='#63b2ee',lw=2)
    plt.title(model_type[model_idx]+' vs +OP vs +SBRT',fontsize=28)    

    plt.xlim(0, 1)
    plt.ylim(0, 25)
    plt.ylabel('Cross-Play Scores', fontsize=24)
    plt.xlabel('CPSTT', fontsize=24)
    plt.legend(fontsize=20, loc='lower right')
    plt.savefig('figs/maincompare'+str(model_idx)+'.pdf', bbox_inches='tight')

#4-for new models
ssd_mean = new_models_cp_mean[0:10,:]
ssd_sim = new_models_cp_b2a[0:10,:]
print('SAD-base, intercp mean:',np.mean(ssd_mean),np.std(ssd_mean)/30)
ssdop_mean = new_models_cp_mean[10:20,:]
ssdop_sim = new_models_cp_b2a[10:20,:]
print('SAD-op, intercp mean:',np.mean(ssdop_mean),np.std(ssdop_mean)/30)
ssdsbrt_mean = new_models_cp_mean[20:30,:]
ssdsbrt_sim = new_models_cp_b2a[20:30,:]
print('SAD-sbrt, intercp mean:',np.mean(ssdsbrt_mean),np.std(ssdsbrt_mean)/30)

ssd_sp_mean = np.load('ssd_intra/score_mean.npy')
ssd_sp_sim = np.load('ssd_intra/distance_b2a.npy')
print('SAD-base, intracp mean:',np.mean(ssd_sp_mean),np.std(ssd_sp_mean)/7)
ssdop_sp_mean = np.load('ssdop_intra/score_mean.npy')
ssdop_sp_sim = np.load('ssdop_intra/distance_b2a.npy')
print('SAD-op, intracp mean:',np.mean(ssdop_sp_mean),np.std(ssdop_sp_mean)/7)
ssdsbrt_sp_mean = np.load('ssdsbrt_intra/score_mean.npy')
ssdsbrt_sp_sim = np.load('ssdsbrt_intra/distance_b2a.npy')
print('SAD-sbrt, intracp mean:',np.mean(ssdsbrt_sp_mean),np.std(ssdsbrt_sp_mean)/7)

plt.figure()
y_data = np.concatenate((ssd_mean, ssd_sp_mean),axis=1).flatten()
x_data = np.concatenate((ssd_sim, ssd_sp_sim),axis=1).flatten()
y_data = np.delete(y_data,x_data>0.95)
x_data = np.delete(x_data,x_data>0.95)
plt.scatter(x_data, y_data, c='#f8cb7f', alpha=0.6, label='SAD')
k,b = np.polyfit(x_data,y_data,1)
x = np.linspace(0,1,50)
y = k*x + b
plt.plot(x,y,c='#f8cb7f',lw=2)

y_data = np.concatenate((ssdop_mean, ssdop_sp_mean),axis=1).flatten()
x_data = np.concatenate((ssdop_sim, ssdop_sp_sim),axis=1).flatten()
y_data = np.delete(y_data,x_data>0.95)
x_data = np.delete(x_data,x_data>0.95)
plt.scatter(x_data, y_data, c='#76da91', alpha=0.6, label='SAD+OP')
k,b = np.polyfit(x_data,y_data,1)
x = np.linspace(0,1,50)
y = k*x + b
plt.plot(x,y,c='#76da91',lw=2)

y_data = np.concatenate((ssdsbrt_mean, ssdsbrt_sp_mean),axis=1).flatten()
x_data = np.concatenate((ssdsbrt_sim, ssdsbrt_sp_sim),axis=1).flatten()
y_data = np.delete(y_data,x_data>0.95)
x_data = np.delete(x_data,x_data>0.95)
plt.scatter(x_data, y_data, c='#63b2ee', alpha=0.6, label='SAD+SBRT')
k,b = np.polyfit(x_data,y_data,1)
x = np.linspace(0,1,50)
y = k*x + b
plt.plot(x,y,c='#63b2ee',lw=2)

plt.title('SAD vs +OP vs +SBRT',fontsize=28)    

plt.xlim(0, 1)
plt.ylim(0, 25)
plt.ylabel('Cross-Play Scores', fontsize=24)
plt.xlabel('CPSTT', fontsize=24)
plt.legend(fontsize=20, loc='lower right')
plt.savefig('figs/maincompare3'+'.pdf', bbox_inches='tight')

#————————————————————————————————————————————————————————————————————————————————————————————#


'''
#below is parameter sharing
score_1 = np.load('base_4_methods_10_seeds/score_mean.npy')[0:10,0:30].flatten()
similarity_1 = np.load('base_4_methods_10_seeds/distance_b2a.npy')[0:10,0:30].flatten()

score_2 = np.load('nosharing_total/score_mean.npy')[:,0:30].flatten()
similarity_2 = np.load('nosharing_total/distance_b2a.npy')[:,0:30].flatten()

label_set = ['IQL with PS','IQL without PS']
score_set = [score_1,score_2]
similarity_set = [similarity_1,similarity_2]
fig_title = 'Impact of Parameter Sharing'
file_name = 'parameter_choose'

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
    plt.plot(x,y,c=color_set[idx],lw=2)
plt.title(fig_title,fontsize=22)    

plt.xlim(0, 1)
plt.ylim(0, 25)
plt.ylabel('Cross-Play Scores', fontsize=18)
plt.xlabel('CPSTT', fontsize=18)
plt.legend(fontsize=14, loc='lower right')
plt.savefig('figs/'+file_name+'.pdf', bbox_inches='tight')





#————————————————————————————————————————————————————————————————————————————————————————————#
#below are 9 figures
base_score = np.load('base_4_methods_10_seeds/score_mean.npy')
base_sim = np.load('base_4_methods_10_seeds/distance_b2a.npy')

iql_score = base_score[0:10,0:30].flatten()
iql_sim = base_sim[0:10,0:30].flatten()

vdn_score = base_score[10:20,0:30].flatten()
vdn_sim = base_sim[10:20,0:30].flatten()

sad_score = base_score[20:30,0:30].flatten()
sad_sim = base_sim[20:30,0:30].flatten()


name = ['t1r03_iql','t2r015_iql','t3r02_iql','t1r03_vdn','t2r015_vdn','t3r02_vdn','t1r02_sad','t2r015_sad','t3r02_sad']
base_scores = [iql_score,vdn_score,sad_score]
base_sims = [iql_sim,vdn_sim,sad_sim]

iql_a_score = np.load('base_4_methods_10_seeds/score_mean.npy')

model_names = ['IQL','VDN','AUX']
type_names = ['SBRT(A)','SBRT(B)','SBRT(C)']

for model_idx in range(3):
    for type_idx in range(3):
        base_name = model_names[model_idx]
        compared_name = base_name +"+"+type_names[type_idx]
        label_set = [base_name,compared_name]
        model_name = name[model_idx*3+type_idx]
        score_new = np.load(model_name+'/score_mean.npy')[:,0:30].flatten()
        sim_new = np.load(model_name+'/distance_b2a.npy')[:,0:30].flatten()

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
        plt.xlabel('CPSTT', fontsize=18)
        plt.legend(fontsize=14, loc='lower right')
        plt.savefig('figs/'+file_name+'.pdf', bbox_inches='tight')

#————————————————————————————————————————————————————————————————————————————————————————————#
#below are alphachoose

score_1 = np.load('base_4_methods_10_seeds/score_mean.npy')[0:10,0:30]
siimlarity_1 = np.load('base_4_methods_10_seeds/distance_b2a.npy')[0:10,0:30]

score_2 = np.load('t3r02_iql/score_mean.npy')[:,0:30]
similarity_2 = np.load('t3r02_iql/distance_b2a.npy')[:,0:30]

score_3 = np.load('t3r04_iql/score_mean.npy')[:,0:30]
similarity_3 = np.load('t3r04_iql/distance_b2a.npy')[:,0:30]

label_set = [r"$\alpha_r$=1",r"$\alpha_r$=0.8",r"$\alpha_r$=0.6"]
score_set = [score_1,score_2,score_3]
similarity_set = [similarity_1,similarity_2,similarity_3]
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








score_2 = np.load('t3r02_vdn/score_mean.npy')[:,0:30]
similarity_2 = np.load('t3r02_vdn/distance_b2a.npy')[:,0:30]

score_3 = np.load('sad_vdn_r04/score_mean.npy')[10:,0:30]
similarity_3 = np.load('sad_vdn_r04/distance_b2a.npy')[10:,0:30]

label_set = [r"$\alpha_r$=1",r"$\alpha_r$=0.8",r"$\alpha_r$=0.6"]
score_set = [score_1,score_2,score_3]
similarity_set = [similarity_1,similarity_2,similarity_3]
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




score_2 = np.load('t3r02_sad/score_mean.npy')[:,0:30]
similarity_2 = np.load('t3r02_sad/distance_b2a.npy')[:,0:30]

score_3 = np.load('sad_vdn_r04/score_mean.npy')[0:10,0:30]
similarity_3 = np.load('sad_vdn_r04/distance_b2a.npy')[0:10,0:30]

label_set = [r"$\alpha_r$=1",r"$\alpha_r$=0.8",r"$\alpha_r$=0.6"]
score_set = [score_1,score_2,score_3]
similarity_set = [similarity_1,similarity_2,similarity_3]
fig_title = 'AUX+SBRT'
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
'''




