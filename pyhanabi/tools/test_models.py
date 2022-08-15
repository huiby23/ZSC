import argparse
import numpy as np
import os
import sys
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model, get_similarities


parser = argparse.ArgumentParser()
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=99999, type=int)
parser.add_argument("--op_type", default=0, type=int)

args = parser.parse_args()

print('first time!', args.op_type)
#args setting here
model_name_a = ['vdn','vdn_sad_aux','vdn_sad_aux_op','iql']
model_name_b = [1,111,1111,11111]
device = 'cuda:0'
agent_a_set = []
agent_b_set = []
exp_name = "base_4_methods_5_seeds"


#if a model exists both in a_set and b_set, it must be at the same location

for type_idx in range(4):
    for seed_idx in range(1,6):
        name = model_name_a[type_idx]+'_seed'+str(seed_idx*model_name_b[type_idx])
        agent_a_set.append(name)
        agent_b_set.append(name)

agent_a_set_num = len(agent_a_set)
agent_b_set_num = len(agent_b_set)

if args.op_type == 0:
    print('second time!', args.op_type)
    #firstly, generate running record
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    if args.start == 0:
        #record model names
        with open(exp_name+'/test_models.txt','w') as f:
            f.write('Set A Models:')
            f.write('\n')
            for string in agent_a_set:
                f.write(string)
                f.write('\n')
            f.write('Set B Models:')
            f.write('\n')
            for string in agent_b_set:
                f.write(string)
                f.write('\n')

        score_mean_matrix = np.zeros((agent_a_set_num, agent_b_set_num))
        score_std_matrix = np.zeros((agent_a_set_num, agent_b_set_num))

    else:
        score_mean_matrix = np.load(exp_name+ "/score_mean.npy")
        score_std_matrix = np.load(exp_name+ "/score_std.npy")

    now_idx = -1

    for a_idx in range(agent_a_set_num):
        for b_idx in range(agent_b_set_num):
            now_idx += 1
            if now_idx < args.start:
                continue
            if now_idx >= args.end:
                continue
            if score_mean_matrix[a_idx,b_idx] > 0:
                continue
            models = ['exps/'+agent_a_set[a_idx]+'/model0.pthw', 'exps/'+agent_b_set[b_idx]+'/model0.pthw', 'exps/'+agent_b_set[b_idx]+'/model0.pthw', 'exps/'+agent_a_set[a_idx]+'/model0.pthw']
            record_path = agent_a_set[a_idx]+'_vs_'+agent_b_set[b_idx]
            if not os.path.exists('../templogs/'+record_path):
                os.makedirs('../templogs/'+record_path)
            score = evaluate_saved_model(models, 1000, 1, 0, device=device, record_name=record_path+'/result')[0]
            score_mean, score_std = np.mean(score), np.std(score)
            score_mean_matrix[a_idx,b_idx] = score_mean
            score_std_matrix[a_idx,b_idx] = score_std
            #check if this record needs to be replicated
            if a_idx!=b_idx:
                if (agent_a_set[a_idx] == agent_b_set[a_idx]) and (agent_a_set[b_idx] == agent_b_set[b_idx]):
                    score_mean_matrix[b_idx,a_idx] = score_mean
                    score_std_matrix[b_idx,a_idx] = score_std 
            print(record_path," completed")           

    #store npy
    np.save(exp_name+ "/score_mean.npy", score_mean_matrix)
    np.save(exp_name+ "/score_std.npy", score_std_matrix)
else:
    print('third time!', args.op_type)
    #secondly, load record and get similarities
    if args.start == 0:
        distance_a2b_matrix = np.zeros((agent_a_set_num, agent_b_set_num))
        distance_b2a_matrix = np.zeros((agent_a_set_num, agent_b_set_num))
    else:
        distance_a2b_matrix = np.load(exp_name+ "/distance_a2b.npy")
        distance_b2a_matrix = np.load(exp_name+ "/distance_b2a.npy")
    now_idx = -1
    for a_idx in range(agent_a_set_num):
        for b_idx in range(agent_b_set_num):
            now_idx += 1
            if now_idx < args.start:
                continue
            if now_idx >= args.end:
                continue
            if distance_a2b_matrix[a_idx,b_idx] > 0:
                continue
            record_name = agent_a_set[a_idx]+'_vs_'+agent_b_set[b_idx]+'/result'
            a2b,b2a = get_similarities(record_name, 10)
            distance_a2b_matrix[a_idx,b_idx] = a2b
            distance_b2a_matrix[a_idx,b_idx] = b2a
            if (agent_a_set[a_idx] == agent_b_set[a_idx]) and (agent_a_set[b_idx] == agent_b_set[b_idx]):
                distance_a2b_matrix[b_idx,a_idx] = b2a
                distance_b2a_matrix[b_idx,a_idx] = a2b      

    #store npy
    np.save(exp_name+ "/distance_a2b.npy", distance_a2b_matrix)
    np.save(exp_name+ "/distance_b2a.npy", distance_b2a_matrix)
    print('analysis data saved')


