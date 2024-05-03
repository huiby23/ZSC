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
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
####################### parameters setting #######################

show_xp = True
show_1zsc = True

# define protagonist set here
protagonist_raw = ['obl_seed1','obl_seed2','obl_seed3','obl_seed4','obl_seed5']
protagonist_pathset = []
for model_name in protagonist_raw:
    protagonist_pathset.append('obl_models/'+model_name+'/model0.pthw')

# define partner set here
partner_pathset = []
name_set = ['iql','vdn','sad','aux']
for type_idx in range(4):
    for seed_idx in range(10):
        partner_pathset.append('encoding_models/'+name_set[type_idx]+'/seed'+str(seed_idx)+'/model0.pthw')

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
                    test_models = [prot_path, part_path]
                    score, _, _, _, _ = evaluate_saved_model(test_models, 500, 0, 0)
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
                    test_models = [prot_path, part_path]
                    score, _, _, _, _ = evaluate_saved_model(test_models, 500, 0, 0)
                    score_set.append(score)
            sys.stdout = original_stdout
        print("Single Score: {:.2f}\pm{:.2f}".format(np.mean(score_set),np.std(score_set)))
        final_results.append(np.mean(score_set))
    print("Final Score: {:.2f}\pm{:.2f}".format(np.mean(final_results),np.std(final_results)))