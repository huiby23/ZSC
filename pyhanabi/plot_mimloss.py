# This script is used to plot the training curves of PBL agents
import pickle
import matplotlib.pyplot as plt
import numpy as np
logs_path_set = []
figs_path_set = []
for idx, log_path in enumerate(logs_path_set):
    with open(log_path,'rb') as f:
        plt.figure()
        log_dict = pickle.load(f)
        x = np.arange(log_dict['score_mm'].shape[0])
        fig, axs = plt.subplots(nrows=1, ncols=2)  
        
        axs[0].plot(x, log_dict['score_mm'],label='main agent self-play') 
        axs[0].plot(x, log_dict['score_mp'],label='cross-play')  
        axs[0].plot(x, log_dict['score_pp'],label='partner agent self-play')  
        axs[0].legend()
        axs[0].set_title('Test Score Curves')  
        axs[0].set_xlabel('epochs')  
        axs[0].set_ylabel('scores')  

        axs[1].plot(x, log_dict['main_rl_loss'],label='main agent rl loss') 
        axs[1].plot(x, log_dict['partner_rl_loss'],label='partner agent rl loss')  
        extraloss = log_dict['partner_extra_loss']
        finalloss = log_dict['partner_rl_loss'] - extraloss
        axs[1].set_ylabel('loss') 
        axs[1].set_xlabel('epochs') 
        axs[1].plot(x, finalloss,label='partner agent total loss')  
        axs[1].twinx() 
        axs[1].plot(x, extraloss, 'r-', label='partner agent extra loss') 
        axs[1].set_ylabel('extra loss')
        axs[1].legend()
        axs[1].set_title('Training Loss Curves')  

        plt.tight_layout()      
    plt.savefig(figs_path_set[idx])    
