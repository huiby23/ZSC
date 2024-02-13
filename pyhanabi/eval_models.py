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

####################### parameters setting #######################


# define protagonist set here

protagonist_pathset = ['ps10_mir0_seed0','ps10_rp_mir0_seed0','ps10_rp_mir003_seed0','ps10_rp_mir01_seed0','ps10_rp_mirsub01_seed0','ps5_mir0_seed0','ps5_rp_mir0_seed0','ps5_rp_mir003_seed0','ps5_rp_mir01_seed0','ps5_rp_mirsub01_seed0']

# define partner set here
partner_pathset = []
name_set = ['iql','vdn','sad','aux']

for type_idx in range(4):
    for seed_idx in range(10):
        partner_pathset.append(name_set[type_idx]+'/seed'+str(seed_idx))

####################### experiment codes #######################
torch.backends.cudnn.benchmark = True

for prot_id, prot_path in enumerate(protagonist_pathset):
    score_set = []
    with open('verbose_out.txt', 'w') as f: # ignore verbose information
        original_stdout = sys.stdout
        sys.stdout = f
        for part_id, part_path in enumerate(partner_pathset):
            test_models = ['models/'+prot_path+'/model0.pthw','models/'+part_path+'/model0.pthw']
            score, _, _, _, _ = evaluate_saved_model(test_models, 500, 0, 0)
            score_set.append(score)
        sys.stdout = original_stdout
    print('prot id: ', prot_id, 'test complete, score:', np.mean(score_set), '+-', np.std(score_set))
