import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--record_name", default=None, type=str, required=True)
parser.add_argument("--thread_num", default=10, type=int, required=False)

args = parser.parse_args()

main_1_actions = []
partner_1_actions = []
main_2_actions = []
partner_2_actions = []
for idx in range(args.thread_num):
    record_name = "../templogs/"+args.record_name+"_"+str(idx)
    main_1_action = np.loadtxt("../templogs/"+record_name+"_1m.txt")
    partner_1_action = np.loadtxt("../templogs/"+record_name+"_1p.txt")
    main_2_action = np.loadtxt("../templogs/"+record_name+"_2m.txt")
    partner_2_action = np.loadtxt("../templogs/"+record_name+"_2p.txt")
    main_1_actions.append(main_1_action) 
    partner_1_actions.append(partner_1_action)
    main_2_actions.append(main_2_action)
    partner_2_actions.append(partner_2_action)
        
    os.remove("../templogs/"+record_name+"_1m.txt")
    os.remove("../templogs/"+record_name+"_1p.txt")
    os.remove("../templogs/"+record_name+"_2m.txt")
    os.remove("../templogs/"+record_name+"_2p.txt")

main_1_actions_total = np.hstack(main_1_actions)
partner_1_actions_total = np.hstack(partner_1_actions)
main_2_actions_total = np.hstack(main_2_actions)
partner_2_actions_total = np.hstack(partner_2_actions)


a_len = main_1_actions_total.shape[0]
a_ratio = np.sum(main_1_actions_total == partner_1_actions_total)/a_len
print("agent_a_similarity:",a_ratio)

b_len = main_2_actions_total.shape[0]
b_ratio = np.sum(main_2_actions_total == partner_2_actions_total)/a_len
print("agent_b_similarity:",b_ratio)