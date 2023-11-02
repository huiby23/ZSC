import argparse
import numpy as np
import os
import sys
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model, get_similarities
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def pearson_coef(x,y):
    X = np.vstack([x,y])
    return np.corrcoef(X)[0][1]

parser = argparse.ArgumentParser()
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=99999, type=int)
parser.add_argument("--op_type", default=0, type=int)
parser.add_argument("--load_100", type=int, default=0)
parser.add_argument("--no_sharing", type=int, default=0)

args = parser.parse_args()


def model_pth(idx):
    model_type_idx = int(idx/10) 
    if model_type_idx % 2 == 1:
        return '/model_epoch100.pthw'
    else:
        return '/model0.pthw'

def jump_judge(now_idx, a_idx, b_idx):
    if (now_idx < args.start) or (now_idx >= args.end):
        return True
    if int(a_idx/20) == int(b_idx/20):
        return True
    if score_mean_matrix[a_idx,b_idx] > 0:
        return True
    return False

#args setting here
model_name_a = ['iql_op','t3r02','vdn_op','t3r02','vdn_sad_aux_op','t3r02']
model_name_b = [11111,11111,1,1,1111,111]
device = 'cuda:0'
agent_a_set = []
agent_b_set = []

exp_name = "old_models_cp"


#if a model exists both in a_set and b_set, it must be at the same location

for type_idx in range(6):
    for seed_idx in range(1,11):
        name = model_name_a[type_idx]+'_seed'+str(seed_idx*model_name_b[type_idx])
        agent_a_set.append(name)
        agent_b_set.append(name)

agent_a_set_num = len(agent_a_set)
agent_b_set_num = len(agent_b_set)

if args.op_type == 0:
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
    else:
        score_mean_matrix = np.load(exp_name+ "/score_mean.npy")

    now_idx = -1

    for a_idx in range(agent_a_set_num):
        for b_idx in range(agent_b_set_num):
            now_idx += 1
            if jump_judge(now_idx, a_idx, b_idx):
                continue

                
            models = ['exps/'+agent_a_set[a_idx]+model_pth(a_idx), 'exps/'+agent_b_set[b_idx]+model_pth(b_idx), 'exps/'+agent_b_set[b_idx]+model_pth(b_idx), 'exps/'+agent_a_set[a_idx]+model_pth(a_idx)]
            record_path = agent_a_set[a_idx]+'_vs_'+agent_b_set[b_idx]

            if not os.path.exists('records/'+record_path):
                os.makedirs('records/'+record_path)
            score = evaluate_saved_model(models, 1000, 1, 0, device=device, record_name=record_path+'/result')[0]
            score_mean = np.mean(score)
            score_mean_matrix[a_idx,b_idx] = score_mean
            #check if this record needs to be replicated
            if a_idx!=b_idx and a_idx<agent_b_set_num and b_idx<agent_a_set_num:
                if (agent_a_set[a_idx] == agent_b_set[a_idx]) and (agent_a_set[b_idx] == agent_b_set[b_idx]):
                    score_mean_matrix[b_idx,a_idx] = score_mean 
            print(record_path," completed") 

    #store npy
    np.save(exp_name+ "/score_mean.npy", score_mean_matrix)
else:
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


