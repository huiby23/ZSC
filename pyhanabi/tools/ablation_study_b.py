import argparse
import numpy as np
import os
import sys
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_path)
from eval import evaluate_saved_model_three
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

parser = argparse.ArgumentParser()
parser.add_argument("--op_type", default=0, type=int)

args = parser.parse_args()

def pearson_coef(x,y):
    X = np.vstack([x,y])
    return np.corrcoef(X)[0][1]

def get_similarities_special(record_name, thread_num=80):
    sim_set = []

    for idx in range(thread_num):
        full_record_name = "records/"+record_name+"_"+str(idx)
        thread_results = np.loadtxt(full_record_name+".txt")
        sim_set.append(thread_results[0]/(thread_results[0]+thread_results[1]))

    return sim_set 

model_names_a = []
model_names_b = []
#args setting here
for idx in range(1,10):
    model_names_a.append('iqlthree_seed'+str(idx*11111))
    model_names_b.append('iqlthree_seed'+str(idx*11111))

device = 'cuda:0'

exp_name = "ablation_b"
if not os.path.exists(exp_name):
    os.makedirs(exp_name)
score_matrix = np.zeros((10,10))
sim_matrix = np.zeros((10,10))

for a_idx in range(10):
    for b_idx in range(10):
        if a_idx == b_idx:
            continue
        models = ['exps/'+model_names_a[a_idx]+'/model0.pthw', 'exps/'+model_names_b[b_idx]+'/model0.pthw']
        record_path = model_names_a[a_idx]+'_vs_'+model_names_b[b_idx]
        if not os.path.exists('records/'+record_path):
            os.makedirs('records/'+record_path)
        score = evaluate_saved_model_three(models, 1000, 1, 0, device=device, record_name=record_path+'/result')[3]
        score_matrix[a_idx,b_idx] = score
        print(record_path," completed")


#store npy
np.save(exp_name+ "/score_mean.npy", score_matrix)
print('score mean: ', score_matrix)


for a_idx in range(10):
    for b_idx in range(10):
        if a_idx == b_idx:
            continue
        record_name = model_names_a[a_idx]+'_vs_'+model_names_b[b_idx]+'/result'
        ratio_set = get_similarities_special(record_name, 10)
        sim_matrix[a_idx,b_idx] = np.mean(ratio_set)

#store npy
np.save(exp_name+ "/distance_mean.npy", sim_matrix)
print('similarity mean: ', sim_matrix) 



score = np.load(exp_name+'/score_mean.npy')
similarity = np.load(exp_name+'/distance_b2a.npy')

fig_title = 'Cross Play Results in 3-Agent Settings'
file_name = 'three_agents'

#plot and analysis
plt.figure()

x_data, y_data = similarity.flatten(), score.flatten()
x_data = np.delete(x_data,y_data==0)
y_data = np.delete(y_data,y_data==0)

plt.scatter(x_data, y_data, c='#f89588', alpha=0.6)
k,b = np.polyfit(x_data,y_data,1)
x = np.linspace(0,1,50)
y = k*x + b
print(' person coef:',pearson_coef(x_data, y_data))
plt.plot(x,y,c='#f89588',lw=2)
plt.title(fig_title,fontsize=22)    

plt.xlim(0, 1)
plt.ylim(0, 25)
plt.ylabel('Cross-Play Scores', fontsize=18)
plt.xlabel('Conditional Policy Similarity', fontsize=18)
#plt.legend(fontsize=14, loc='lower right')
plt.savefig('figs/'+file_name+'.pdf', bbox_inches='tight')