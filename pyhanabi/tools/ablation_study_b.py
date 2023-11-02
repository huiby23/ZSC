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
    partner_1_sim = 0
    partner_1_diff = 0
    partner_2_sim = 0
    partner_2_diff = 0

    for idx in range(thread_num):
        full_record_name = "records/"+record_name+"_"+str(idx)
        thread_results_1 = np.loadtxt(full_record_name+"_part1.txt")
        thread_results_2 = np.loadtxt(full_record_name+"_part2.txt")
        partner_1_sim += thread_results_1[0]
        partner_1_diff += thread_results_1[1]
        partner_2_sim += thread_results_2[0]
        partner_2_diff += thread_results_2[1]

    return partner_1_sim/(partner_1_sim+partner_1_diff), partner_2_sim/(partner_2_sim+partner_2_diff)

model_names_a = []
model_names_b = []
model_names_c = []
#args setting here
for idx in range(1,11):
    model_names_a.append('iqlthree_seed'+str(idx*11111))
    model_names_b.append('iqlthree_seed'+str(idx*11111))
    model_names_c.append('iqlthree_seed'+str(idx*11111))

device = 'cuda:0'

exp_name = "ablation_b_new"

'''
if not os.path.exists(exp_name):
    os.makedirs(exp_name)

score_record = np.zeros(1000)


for a_idx in range(10):
    for b_idx in range(10):
        for c_idx in range(10):
            final_idx = a_idx*100+b_idx*10+c_idx
            models = ['exps/'+model_names_a[a_idx]+'/model0.pthw','exps/'+model_names_b[b_idx]+'/model0.pthw','exps/'+model_names_a[a_idx]+'/model0.pthw','exps/'+model_names_c[c_idx]+'/model0.pthw','exps/'+model_names_a[a_idx]+'/model0.pthw']
            record_path = 'threetest_'+str(final_idx)
            if not os.path.exists('records/'+record_path):
                os.makedirs('records/'+record_path)
            score = evaluate_saved_model_three(models, 1000, 1, 0, device=device, record_name=record_path+'/result')[3]
            score_record[final_idx] = np.mean(score)
            print(record_path," completed")


#store npy
np.save(exp_name+ "/score_mean.npy", score_record)
print('score mean: ', score_record)







score_record = np.zeros((1000,2))
for a_idx in range(10):
    for b_idx in range(10):
        for c_idx in range(10):
            final_idx = a_idx*100+b_idx*10+c_idx
            record_name = 'threetest_'+str(final_idx)+'/result'
            part_1_sim, part_2_sim = get_similarities_special(record_name, 10)
            score_record[final_idx,0] = part_1_sim
            score_record[final_idx,1] = part_2_sim

#store npy
np.save(exp_name+ "/distance_mean.npy", score_record)
print('similarity mean: ', score_record) 

'''



score = np.load(exp_name+'/score_mean.npy')
similarity = np.load(exp_name+'/distance_mean.npy')

fig_title = 'Cross Play Results in 3-Agent Games'
file_name = 'three_agents'

#plot and analysis
plt.figure()

mean_similarity = np.mean(similarity,axis=1)

x_data = []
y_data = []
for a_idx in range(10):
    for b_idx in range(10):
        if b_idx == a_idx:
            continue
        for c_idx in range(10):
            if c_idx == a_idx:
                continue
            elif c_idx == b_idx:
                continue
            else:
                final_idx = a_idx*100+b_idx*10+c_idx
                y_data.append(score[final_idx])
                x_data.append(mean_similarity[final_idx])

y_data = np.array(y_data)
x_data = np.array(x_data)


plt.scatter(x_data, y_data, c='#f89588', alpha=0.6)
k,b = np.polyfit(x_data,y_data,1)
x = np.linspace(0,1,50)
y = k*x + b
print('person coef:',pearson_coef(x_data, y_data))


plt.plot(x,y,c='#f89588',lw=2)
plt.title(fig_title,fontsize=22)    

plt.xlim(0, 1)
plt.ylim(0, 25)
plt.ylabel('Cross-Play Scores', fontsize=18)
plt.xlabel('CPSTT', fontsize=18)
#plt.legend(fontsize=14, loc='lower right')
plt.savefig('figs/'+file_name+'.pdf', bbox_inches='tight')
