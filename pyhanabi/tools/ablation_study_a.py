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
    main_1_actions = []
    partner_1_actions = []
    main_2_actions = []
    partner_2_actions = []
    for idx in range(thread_num):
        full_record_name = "../templogs/"+record_name+"_"+str(idx)
        main_1_action = np.loadtxt(full_record_name+"_1m.txt")
        partner_1_action = np.loadtxt(full_record_name+"_1p.txt")
        main_2_action = np.loadtxt(full_record_name+"_2m.txt")
        partner_2_action = np.loadtxt(full_record_name+"_2p.txt")
        main_1_actions.append(main_1_action) 
        partner_1_actions.append(partner_1_action)
        main_2_actions.append(main_2_action)
        partner_2_actions.append(partner_2_action)
            
        os.remove(full_record_name+"_1m.txt")
        os.remove(full_record_name+"_1p.txt")
        os.remove(full_record_name+"_2m.txt")
        os.remove(full_record_name+"_2p.txt")

    main_1_actions_total = np.hstack(main_1_actions)
    partner_1_actions_total = np.hstack(partner_1_actions)
    main_2_actions_total = np.hstack(main_2_actions)
    partner_2_actions_total = np.hstack(partner_2_actions)

    #divide into 100 pieces
    a_ratio_set = np.zeros(100)
    a_seq_len = int(main_1_actions_total.shape[0]/100)
    b_ratio_set = np.zeros(100)
    b_seq_len = int(main_2_actions_total.shape[0]/100)
    a_sim_record = main_1_actions_total == partner_1_actions_total
    b_sim_record = main_2_actions_total == partner_2_actions_total

    for idx in range(99):
        a_ratio_set[idx] = np.mean(a_sim_record[idx*a_seq_len:(idx+1)*a_seq_len])
        b_ratio_set[idx] = np.mean(b_sim_record[idx*b_seq_len:(idx+1)*b_seq_len])
    a_ratio_set[99] = np.mean(a_sim_record[99*a_seq_len:])
    b_ratio_set[99] = np.mean(b_sim_record[99*b_seq_len:])

    return a_ratio_set, b_ratio_set 


#args setting here
model_names = ['iql_seed11111','vdn_seed1','vdn_sad_aux_seed111','obl_seed11']
device = 'cuda:0'

exp_name = "ablation_a"
if not os.path.exists(exp_name):
    os.makedirs(exp_name)
score_mean_matrix = np.zeros((4,4))
score_std_matrix = np.zeros((4,4))

if args.op_type == 0:
    for a_idx in range(4):
        for b_idx in range(4):
            models = ['exps/'+model_names[a_idx]+'/model0.pthw', 'exps/'+model_names[b_idx]+'/model0.pthw', 'exps/'+model_names[b_idx]+'/model0.pthw', 'exps/'+model_names[a_idx]+'/model0.pthw']
            record_path = model_names[a_idx]+'_vs_'+model_names[b_idx]
            if not os.path.exists('../templogs/'+record_path):
                os.makedirs('../templogs/'+record_path)
            score = evaluate_saved_model(models, 100000, 1, 0, device=device, record_name=record_path+'/result')[3]
            score_lists = np.zeros(100)
            seq_len = int(len(score)/100)
            for idx in range(99):
                score_lists[idx] = np.mean(score[seq_len*idx:seq_len*(idx+1)])
            score_lists[99] = np.mean(score[seq_len*99:])

            score_mean, score_std = np.mean(score_lists), np.std(score_lists)/10
            score_mean_matrix[a_idx,b_idx] = score_mean
            score_std_matrix[a_idx,b_idx] = score_std   

    #store npy
    np.save(exp_name+ "/score_mean.npy", score_mean_matrix)
    np.save(exp_name+ "/score_std.npy", score_std_matrix)
    print('score mean:', score_mean_matrix)
    print('score std:', score_std_matrix)

else:
    #secondly, load record and get similarities
    distance_b2a_matrix_mean = np.zeros((4, 4))
    distance_b2a_matrix_std = np.zeros((4, 4))

    for a_idx in range(4):
        for b_idx in range(4):
            record_name = model_names[a_idx]+'_vs_'+model_names[b_idx]+'/result'
            _,b_ratio_set = get_similarities_special(record_name, 10)
            distance_b2a_matrix_mean[a_idx,b_idx] = np.mean(b_ratio_set)
            distance_b2a_matrix_std[a_idx,b_idx] = np.std(b_ratio_set)  

    #store npy
    np.save(exp_name+ "/distance_mean.npy", distance_b2a_matrix_mean)
    np.save(exp_name+ "/distance_std.npy", distance_b2a_matrix_std)
    print('similarity mean: ', distance_b2a_matrix_mean) 
    print('similarity std: ', distance_b2a_matrix_std) 