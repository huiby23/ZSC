# This script is used to test agent performance

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
from pathlib import Path
base_dir = Path(__file__).resolve().parent
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
####################### parameters setting #######################

show_xp = True
show_1zsc = True
num_player = 5
# define protagonist set here
# protagonist_raw = ['s'+str(x)+'_ps3_div1_mm1_mp2_w1' for x in range(3)]
# protagonist_raw = ['5p_s4_ps5_div2_mm1_mp2_w01']
seed_set=[]
protagonist_raw = ['5p_s2_ps'+str(x)+'_div2_mm1_mp2_w01' for x in [5,8,12,15]]
protagonist_pathset = []
for model_name in protagonist_raw:
    protagonist_pathset.append(os.path.join(base_dir,'subnet_models/'+model_name+'/model0.pthw'))

# define partner set here
partner_pathset = []
# name_set = ['iql','vdn','sad','aux']
name_set = ['iql_5p','vdn_5p','sad_5p','aux_5p']
# for type_idx in range(4):
#     for seed_idx in range(1):
#         partner_pathset.append('models/'+name_set[type_idx]+'/seed'+str(seed_idx)+'/model0.pthw')
for type_agent in name_set:
    for seed_idx in range(10):
        partner_pathset.append(os.path.join(base_dir,'models/'+type_agent+'/seed'+str(seed_idx)+'/model0.pthw'))
####################### experiment codes #######################
torch.backends.cudnn.benchmark = True

if show_1zsc:
    print('1-ZSC test start')
    final_results = []
    for _, prot_path in enumerate(protagonist_pathset):
        score_set = []
        with open('verbose_out.txt', 'w') as f: # ignore verbose information
            original_stdout = sys.stdout
            sys.stdout = f
            for part_id, part_path in enumerate(partner_pathset):
                if prot_path != part_path:
                    test_models = [prot_path]+[part_path]*(num_player-1)
                    score, _, _, _, _ = evaluate_saved_model(test_models, 100, 1000, 0)
                    score_set.append(score)
            sys.stdout = original_stdout
        print("Single Score: {:.2f}\pm{:.2f}".format(np.mean(score_set),np.std(score_set)))
        final_results.append(np.mean(score_set))
    print("Final Score: {:.2f}\pm{:.2f}".format(np.mean(final_results),np.std(final_results)))

if show_xp:
    print('xp test start')
    final_results = []
    for _, prot_path in enumerate(protagonist_pathset):
        score_set = []
        with open('verbose_out.txt', 'w') as f: # ignore verbose information
            original_stdout = sys.stdout
            sys.stdout = f
            for _, part_path in enumerate(protagonist_pathset):
                if prot_path != part_path:
                    test_models = [prot_path]+[part_path]*(num_player-1)
                    score, _, _, _, _ = evaluate_saved_model(test_models, 500, 101, 0)
                    score_set.append(score)
            sys.stdout = original_stdout
        print("Single Score: {:.2f}\pm{:.2f}".format(np.mean(score_set),np.std(score_set)))
        final_results.append(np.mean(score_set))
    print("Final Score: {:.2f}\pm{:.2f}".format(np.mean(final_results),np.std(final_results)))