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

parser = argparse.ArgumentParser()
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--end", default=99999, type=int)
parser.add_argument("--op_type", default=0, type=int)
parser.add_argument("--test_one", default=0, type=int)
parser.add_argument("--test_model_name", default=None, type=str)
parser.add_argument("--ref_record_name", default=None, type=str)

args = parser.parse_args()

#args setting here
model_name_a = ['vdn','vdn_sad_aux','vdn_sad_aux_op','iql']
model_name_b = [1,111,1111,11111]
device = 'cuda:0'
agent_a_set = []
agent_b_set = []
exp_name = "base_4_methods_5_seeds"

def pearson_coef(x,y):
    X = np.vstack([x,y])
    return np.corrcoef(X)[0][1]
#if a model exists both in a_set and b_set, it must be at the same location

for type_idx in range(4):
    for seed_idx in range(1,6):
        name = model_name_a[type_idx]+'_seed'+str(seed_idx*model_name_b[type_idx])
        agent_a_set.append(name)
        agent_b_set.append(name)

if args.test_one == 1:
    agent_a_set = [args.test_model_name]
    exp_name = "test_" + args.test_model_name
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

        #store npy
        np.save(exp_name+ "/score_mean.npy", score_mean_matrix)
        np.save(exp_name+ "/score_std.npy", score_std_matrix)
        print(args.test_model_name, ' score mean:', score_mean_matrix)

    elif args.op_type == 1:
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
        print('similarity: ', distance_a2b_matrix) 
    elif args.op_type == 2: #load npy and analysis
        y_data = np.load(exp_name+ "/score_mean.npy").flatten()
        x_data = np.load(exp_name+ "/distance_b2a.npy").flatten()
        k,b = np.polyfit(x_data,y_data,1)
        p_coef = pearson_coef(x_data, y_data)
        point_1,point_2,point_3 = 0.4*k+b, 0.6*k+b, 0.8*k+b
        print('k:',"%.2f" %k)
        print('b:',"%.2f" %b)
        print('pearson coefficient:', "%.2f" %p_coef)
        print('0.4:',"%.2f"%point_1)
        print('0.6:',"%.2f"%point_2)
        print('0.8:',"%.2f"%point_3)   
    else:#plot
        y_data = np.load(exp_name+ "/score_mean.npy").flatten()
        x_data = np.load(exp_name+ "/distance_a2b.npy").flatten()  

        ref_y_data = np.load(args.ref_record_name + "/score_mean.npy").flatten()
        ref_x_data = np.load(args.ref_record_name + "/distance_b2a.npy").flatten() 
        plt.scatter(x_data, y_data, c='#0082fc', alpha=0.8,label='test model')      
        plt.scatter(ref_x_data, ref_y_data, c='#fdd845', alpha=0.8,label='ref model')
        plt.title(exp_name,fontsize=22)
        plt.xlim(0, 1)
        plt.ylim(0, 25)        
        plt.ylabel('performance', fontsize=18)
        plt.xlabel('similarity', fontsize=18)
        plt.legend(fontsize=14, loc='lower right')
        plt.savefig('figs/model_'+exp_name+'.pdf', bbox_inches='tight')
        print('figure saved!')
elif args.test_one == 2: #test a group of agents
    #note that agent_a_set and agent_b_set must be predefined
    exp_name = "test_" + args.test_model_name
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

        #store npy
        np.save(exp_name+ "/score_mean.npy", score_mean_matrix)
        np.save(exp_name+ "/score_std.npy", score_std_matrix)
        print(args.test_model_name, ' score mean:', score_mean_matrix)

    elif args.op_type == 1:
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
        print('similarity: ', distance_a2b_matrix) 
    elif args.op_type == 2: #load npy and analysis

        y_data = np.load(exp_name+ "/score_mean.npy").flatten()
        x_data = np.load(exp_name+ "/distance_b2a.npy").flatten()
        #remove self-play points
        y_data = np.delete(y_data,x_data>0.95)
        x_data = np.delete(x_data,x_data>0.95)

        k,b = np.polyfit(x_data,y_data,1)
        p_coef = pearson_coef(x_data, y_data)
        point_1,point_2,point_3 = 0.4*k+b, 0.6*k+b, 0.8*k+b
        print('k:',"%.2f" %k)
        print('b:',"%.2f" %b)
        print('pearson coefficient:', "%.2f" %p_coef)
        print('0.4:',"%.2f"%point_1)
        print('0.6:',"%.2f"%point_2)
        print('0.8:',"%.2f"%point_3)   
    else: #load npy and analysis
        y_data = np.load(exp_name+ "/score_mean.npy").flatten()
        x_data = np.load(exp_name+ "/distance_a2b.npy").flatten()  
        #remove self-play points
        y_data = np.delete(y_data,x_data>0.95)
        x_data = np.delete(x_data,x_data>0.95)

        ref_y_data = np.load(args.ref_record_name + "/score_mean.npy").flatten()
        ref_x_data = np.load(args.ref_record_name + "/distance_b2a.npy").flatten() 
        #remove self-play points
        ref_y_data = np.delete(ref_y_data,ref_x_data>0.95)
        ref_x_data = np.delete(ref_x_data,ref_x_data>0.95)

        plt.scatter(x_data, y_data, c='#7cd6cf', alpha=0.8,label='test model')      
        plt.scatter(ref_x_data, ref_y_data, c='#f89588', alpha=0.8,label='ref model')
        plt.title(exp_name, fontsize=22)
        plt.xlim(0, 1)
        plt.ylim(0, 25)        
        plt.ylabel('performance', fontsize=18)
        plt.xlabel('similarity', fontsize=18)
        plt.legend(fontsize=14, loc='lower right')
        plt.savefig('figs/model_'+exp_name+'.pdf', bbox_inches='tight')
        print('figure saved!')


else:
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


