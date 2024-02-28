# This script is used to plot the training curves of PBL agents
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve 
logs_path_set = ['ps5_seed0','new_ps5_div2_trn2_w005','new_ps3_div0_trn2_w005','new_ps5_div1_trn2_w0025']
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