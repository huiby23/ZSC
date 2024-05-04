# This script is used to plot the training curves of PBL agents
# Style 1: One figure, one model
import pickle
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from scipy.ndimage import convolve 
#logs_path_set = ['ps5_seed0','new_ps5_div2_trn2_w005','new_ps3_div0_trn2_w005','new_ps5_div1_trn2_w0025']

model_path_set = ['s'+str(i)+'_ps5_div1_mp1_w1' for i in range(1,6)]
logs_path_set = ['subnet_models/'+log_path+'/train_log.pkl' for log_path in model_path_set]

figure_title = 'Type-I Training'
save_path = 'figs/type1.pdf'

score_mm = []
score_mp = []
score_pp = []

for idx, log_path in enumerate(logs_path_set):
    with open(log_path,'rb') as f:
        log_dict = pickle.load(f)
        score_mm.append(log_dict['score_mm'])
        score_mp.append(log_dict['score_mp'])
        score_pp.append(log_dict['score_pp'])

score_mm = np.vstack(score_mm)
score_mp = np.vstack(score_mp)
score_pp = np.vstack(score_pp)
plt.figure()

x = np.arange(score_mm.shape[1])

dataset = [score_mm,score_mp,score_pp]
label_set = ['MM score', 'MP score', 'PP score']
color_set = ['#194f97','#bd6b08','#c82d31']

for idx in range(3):
    datamean, dataste = dataset[idx].mean(axis=0), dataset[idx].std(axis=0)/2.3
    plt.plot(x, datamean, label=label_set[idx], color=color_set[idx])
    plt.fill_between(x, datamean-dataste, datamean+dataste, alpha=0.2, color=color_set[idx])
plt.ylim(0,24)
plt.yticks(np.arange(0,27,3))
plt.title(figure_title,fontsize=18)  
plt.xlabel('Epoch',fontsize=16)  
#plt.ylabel('Score',fontsize=16)
plt.legend(fontsize=14,loc='upper left')
plt.tight_layout()      
plt.savefig(save_path)    

'''
label_set = ['Baseline','PEM','CMIM-A','CMIM-B']
save_name = 'figs/main'
window_size = 5  
weights = np.ones(window_size) / window_size  

plt.figure()
for idx, log_path in enumerate(logs_path_set):
    target_path = 'models/'+log_path+'/train_log.pkl'
    with open(target_path,'rb') as f:
        log_dict = pickle.load(f)
        x = np.arange(log_dict['score_mm'].shape[0])
        plt.plot(x, convolve(log_dict['score_mm'], weights, mode='nearest'),label=label_set[idx]) 

plt.title('Main vs Main scores')  
plt.xlabel('Epoch')  
plt.legend()
plt.tight_layout()      
plt.savefig(save_name+'_main.pdf')    

plt.figure()
for idx, log_path in enumerate(logs_path_set):
    target_path = 'models/'+log_path+'/train_log.pkl'
    with open(target_path,'rb') as f:
        log_dict = pickle.load(f)
        x = np.arange(log_dict['score_mp'].shape[0])
        plt.plot(x, convolve(log_dict['score_mp'], weights, mode='nearest'),label=label_set[idx]) 

plt.title('Main vs Partner scores')  
plt.xlabel('Epoch')  
plt.legend()
plt.tight_layout()      
plt.savefig(save_name+'_part.pdf')   

'''
 