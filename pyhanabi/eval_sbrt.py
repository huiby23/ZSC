# Note: the following codes are used to eval group performance

import time
import os
import sys
import pprint
import pickle
import numpy as np
import torch
from torch import nn

from act_group import ActGroup
from create import create_envs, create_threads
from eval import evaluate, evaluate_saved_model
import common_utils
import rela
import r2d2
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
torch.backends.cudnn.benchmark = True
def evaluate_group(protagonist_pathset, partner_pathset, num_episodes, prota_sbrt=False, part_sbrt=False):
    score_set = []
    with open('verbose_out.txt', 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        for prot_path in protagonist_pathset:
            for part_path in partner_pathset:
                if prot_path != part_path:
                    if prota_sbrt:
                        model_a = 'models/'+prot_path+'/model_epoch100.pthw'
                    else:
                        model_a = 'models/'+prot_path+'/model0.pthw'
                    if part_sbrt:
                        model_b = 'models/'+part_path+'/model_epoch100.pthw'
                    else:
                        model_b = 'models/'+part_path+'/model0.pthw'
                    test_models = [model_a, model_b]
                    score, _, _, _, _ = evaluate_saved_model(test_models, num_episodes, 0, 0)
                    score_set.append(score)
        sys.stdout = original_stdout
    return np.mean(score_set), np.std(score_set)/np.sqrt(len(score_set))

####################### parameters setting #######################
# define agent set here

base_pathset = []
op_pathset = []
sbrt1_pathset = []
sbrt2_pathset = []
sbrt3_pathset = []
base_nameset = ['iql','vdn','sad','aux']
for type_idx in range(4):
    for seed_idx in range(10):
        base_pathset.append(base_nameset[type_idx]+'/seed'+str(seed_idx))
        op_pathset.append(base_nameset[type_idx]+'_op/seed'+str(seed_idx))
        sbrt1_pathset.append(base_nameset[type_idx]+'_sbrt/at1_seed'+str(seed_idx))
        sbrt2_pathset.append(base_nameset[type_idx]+'_sbrt/at2_seed'+str(seed_idx))
        sbrt3_pathset.append(base_nameset[type_idx]+'_sbrt/at3_seed'+str(seed_idx))

# test iql variants performance 
print('IQL variants performance')
print('Base inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(base_pathset[0:10], base_pathset[0:10], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('Base inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(base_pathset[0:10], base_pathset[10:], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('OP inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(op_pathset[0:10], op_pathset[0:10], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('OP inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(op_pathset[0:10], base_pathset[10:], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT1 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt1_pathset[0:10], sbrt1_pathset[0:10], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT1 inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt1_pathset[0:10], base_pathset[10:], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT2 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt2_pathset[0:10], sbrt2_pathset[0:10], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT2 inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt2_pathset[0:10], base_pathset[10:], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT3 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt3_pathset[0:10], sbrt3_pathset[0:10], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT3 inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt3_pathset[0:10], base_pathset[10:], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)


# test vdn variants performance 
print('VDN variants performance')
print('Base inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(base_pathset[10:20], base_pathset[10:20], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('Base inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(base_pathset[10:20], base_pathset[:10]+base_pathset[20:], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('OP inner-algorithm performance:',end=' ') 
mean_score, std_score = evaluate_group(op_pathset[10:20], op_pathset[10:20], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('OP inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(op_pathset[10:20], base_pathset[:10]+base_pathset[20:], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT1 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt1_pathset[10:20], sbrt1_pathset[10:20], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT1 inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt1_pathset[10:20], base_pathset[:10]+base_pathset[20:], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT2 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt2_pathset[10:20], sbrt2_pathset[10:20], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT2 inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt2_pathset[10:20], base_pathset[:10]+base_pathset[20:], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT3 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt3_pathset[10:20], sbrt3_pathset[10:20], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT3 inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt3_pathset[10:20], base_pathset[:10]+base_pathset[20:], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)


# test sad variants performance 
print('SAD variants performance')
print('Base inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(base_pathset[20:30], base_pathset[20:30], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('Base inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(base_pathset[20:30], base_pathset[:20]+base_pathset[30:], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('OP inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(op_pathset[20:30], op_pathset[20:30], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('OP inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(op_pathset[20:30], base_pathset[:20]+base_pathset[30:], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT1 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt1_pathset[20:30], sbrt1_pathset[20:30], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT1 inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt1_pathset[20:30], base_pathset[:20]+base_pathset[30:], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT2 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt2_pathset[20:30], sbrt2_pathset[20:30], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT2 inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt2_pathset[20:30], base_pathset[:20]+base_pathset[30:], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT3 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt3_pathset[20:30], sbrt3_pathset[20:30], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT3 inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt3_pathset[20:30], base_pathset[:20]+base_pathset[30:], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)


# test aux variants performance 
print('AUX variants performance')
print('Base inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(base_pathset[30:40], base_pathset[30:40], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('Base inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(base_pathset[30:40], base_pathset[:30], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('OP inner-algorithm performance:',end=' ') 
mean_score, std_score = evaluate_group(op_pathset[30:40], op_pathset[30:40], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('OP inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(op_pathset[30:40], base_pathset[:30], 1000)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT1 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt1_pathset[30:40], sbrt1_pathset[30:40], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT1 inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt1_pathset[30:40], base_pathset[:30], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT2 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt2_pathset[30:40], sbrt2_pathset[30:40], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT2 inter-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt2_pathset[30:40], base_pathset[:30], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT3 inner-algorithm performance:',end=' ')
mean_score, std_score = evaluate_group(sbrt3_pathset[30:40], sbrt3_pathset[30:40], 1000, prota_sbrt=True,part_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
print('SBRT3 inter-algorithm performance:',end=' ') 
mean_score, std_score = evaluate_group(sbrt3_pathset[30:40], base_pathset[:30], 1000, prota_sbrt=True)
print('Mean:', mean_score, 'Std:', std_score)
