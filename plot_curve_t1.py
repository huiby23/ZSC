# This script is used to plot the training curves of PBL agents
# Style 1: One figure, one model
import pickle
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from scipy.ndimage import convolve 
#logs_path_set = ['ps5_seed0','new_ps5_div2_trn2_w005','new_ps3_div0_trn2_w005','new_ps5_div1_trn2_w0025']

model_path_set = ['s'+str(i)+'_ps5_div1_mm1_mp2_w1' for i in range(1,6)]
logs_path_set = ['subnet_models/'+log_path+'/train_log.pkl' for log_path in model_path_set]

figure_title = r'$\alpha$=1'
save_path = 'figs/w1.pdf'

score_mm = []
score_mp = []
score_pp = []
extra_info = []

for idx, log_path in enumerate(logs_path_set):
    with open(log_path,'rb') as f:
        log_dict = pickle.load(f)
        score_mm.append(log_dict['score_mm'])
        score_mp.append(log_dict['score_mp'])
        score_pp.append(log_dict['score_pp'])
        extra_info.append(log_dict['partner_extra_info']/0.8)

score_mm = np.vstack(score_mm)
score_mp = np.vstack(score_mp)
score_pp = np.vstack(score_pp)
extra_info = np.vstack(extra_info)


x = np.arange(score_mm.shape[1])

dataset = [score_mm,score_mp,score_pp]
label_set = ['MM Score', 'MP Score', 'PP Score']
color_set = ['#194f97','#bd6b08','#c82d31']

fig = plt.figure()
ax1 = fig.add_subplot(111)

for idx in range(3):
    datamean, dataste = dataset[idx].mean(axis=0), dataset[idx].std(axis=0)/2.3
    ax1.plot(x, datamean, label=label_set[idx], color=color_set[idx],lw=2)
    ax1.fill_between(x, datamean-dataste, datamean+dataste, alpha=0.2, color=color_set[idx])
datamean, dataste = extra_info.mean(axis=0), extra_info.std(axis=0)/2.3
ax2 = ax1.twinx()
ax2.plot(x, datamean, label='Diff Prob', color='#625ba1',lw=2)
ax2.fill_between(x, datamean-dataste, datamean+dataste, alpha=0.2, color='#625ba1')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.set_ylabel('Score',fontsize=18)
ax2.set_ylabel('Probability',fontsize=18)
ax1.set_xlabel('Epoch',fontsize=18)
ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)
ax1.set_ylim(0,24)
ax1.set_yticks(np.arange(0,27,3))
ax2.set_ylim(0,1)
ax2.set_yticks(np.arange(0,1.2,0.2))

ax1.set_title(figure_title,fontsize=18)    
plt.savefig(save_path,bbox_inches='tight') 

'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
bar1 = ax1.bar(x1-0.5*bar_width, thb_steps,bar_width,bottom = 0, label='时间步', ec='black',color='#7cd6cf')
ax2 = ax1.twinx()
bar2 = ax2.bar(x2+0.5*bar_width, thb_ent,bar_width,bottom = 0, label='熵', ec='black', color='#9192ab')
ax1.set_ylabel('时间步↓',fontsize=18)
ax2.set_ylabel('熵↓',fontsize=18)
ax1.set_xlabel('$\Delta$',fontsize=18)
ax1.set_xticks(ticks=x1)
ax1.set_xticklabels(xticks)
lns = bar1 + bar2
labs = [l.get_label() for l in lns]
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='lower left',fontsize=16)

ax1.set_title('寻宝-II',fontsize=18)
#plt.show()
plt.savefig('C:/Figures_Thesis/research_01/thb_quant.pdf',bbox_inches='tight')


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
 