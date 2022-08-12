import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--record_name", default=None, type=str, required=True)

args = parser.parse_args()

agent_a_data = np.load("../dataset/"+args.record_name+"_a.npy")
a_len = agent_a_data.shape[1]
a_ratio = np.sum(agent_a_data[0,:] == agent_a_data[1,:])/a_len
print("agent_a_similarity:",a_ratio)

agent_b_data = np.load("../dataset/"+args.record_name+"_a.npy")
b_len = agent_b_data.shape[1]
b_ratio = np.sum(agent_b_data[0,:] == agent_b_data[1,:])/a_len
print("agent_b_similarity:",b_ratio)