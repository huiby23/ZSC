import argparse
import numpy as np
import os
import sys
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

parser = argparse.ArgumentParser()
parser.add_argument("--op_type", default=0, type=int)

args = parser.parse_args()

def get_similarities_special(record_name, thread_num=80):
    sim_set_a = []
    sim_set_b = []

    for idx in range(thread_num):
        full_record_name = "records/"+record_name+"_"+str(idx)
        thread_results = np.loadtxt(full_record_name+".txt")
        sim_set_a.append(thread_results[0]/(thread_results[0]+thread_results[1]))
        sim_set_b.append(thread_results[2]/(thread_results[2]+thread_results[3]))

    return sim_set_a, sim_set_b 


#args setting here
model_names_a = ['iql_seed11111','vdn_seed1','vdn_sad_aux_seed111','obl_seed11']
model_names_b = ['iql_seed22222','vdn_seed2','vdn_sad_aux_seed222','obl_seed22']
device = 'cuda:0'

exp_name = "ablation_a"
if not os.path.exists(exp_name):
    os.makedirs(exp_name)
score_mean_matrix = np.zeros((4,4))
score_std_matrix = np.zeros((4,4))

for a_idx in range(4):
    for b_idx in range(4):
        models = ['exps/'+model_names_a[a_idx]+'/model0.pthw', 'exps/'+model_names_b[b_idx]+'/model0.pthw', 'exps/'+model_names_b[b_idx]+'/model0.pthw', 'exps/'+model_names_a[a_idx]+'/model0.pthw']
        record_path = model_names_a[a_idx]+'_vs_'+model_names_b[b_idx]
        if not os.path.exists('../templogs/'+record_path):
            os.makedirs('../templogs/'+record_path)
        score = evaluate_saved_model(models, 100000, 1, 0, device=device, record_name=record_path+'/result')[3]
        score_lists = np.zeros(10)
        seq_len = int(len(score)/10)
        for idx in range(9):
            score_lists[idx] = np.mean(score[seq_len*idx:seq_len*(idx+1)])
        score_lists[9] = np.mean(score[seq_len*9:])

        score_mean, score_std = np.mean(score_lists), np.std(score_lists)
        score_mean_matrix[a_idx,b_idx] = score_mean
        score_std_matrix[a_idx,b_idx] = score_std   

#store npy
np.save(exp_name+ "/score_mean.npy", score_mean_matrix)
np.save(exp_name+ "/score_std.npy", score_std_matrix)
print('score mean:', score_mean_matrix)
print('score std:', score_std_matrix)

#secondly, load record and get similarities
distance_b2a_matrix_mean = np.zeros((4, 4))
distance_b2a_matrix_std = np.zeros((4, 4))

for a_idx in range(4):
    for b_idx in range(4):
        record_name = model_names_a[a_idx]+'_vs_'+model_names_b[b_idx]+'/result'
        _,b_ratio_set = get_similarities_special(record_name, 10)
        distance_b2a_matrix_mean[a_idx,b_idx] = np.mean(b_ratio_set)
        distance_b2a_matrix_std[a_idx,b_idx] = np.std(b_ratio_set)  

#store npy
np.save(exp_name+ "/distance_mean.npy", distance_b2a_matrix_mean)
np.save(exp_name+ "/distance_std.npy", distance_b2a_matrix_std)
print('similarity mean: ', distance_b2a_matrix_mean) 
print('similarity std: ', distance_b2a_matrix_std) 