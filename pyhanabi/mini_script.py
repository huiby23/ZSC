# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np

def pearson_coef(x,y):
    X = np.vstack([x,y])
    return np.corrcoef(X)[0][1]

#A comprehensive analysis of base test 

base_scores = np.load('base_4_methods_10_seeds/score_mean.npy')
base_similarity = np.load('base_4_methods_10_seeds/distance_b2a.npy')

#Firstly, vdn
vdn_intra_scores = base_scores[10:20,10:20].flatten()
vdn_intra_sim = base_similarity[10:20,10:20].flatten()
vdn_intra_scores = np.delete(vdn_intra_scores,vdn_intra_sim>0.95)
vdn_inter_scores = np.hstack([base_scores[10:20,0:10],base_scores[10:20,20:]]).flatten()

vdn_op_intra_scores = np.load('vdn_op_intraplay/score_mean.npy').flatten()
vdn_op_intra_sim = np.load('vdn_op_intraplay/distance_b2a.npy').flatten()
vdn_op_intra_scores = np.delete(vdn_op_intra_scores, vdn_op_intra_sim>0.95)
total_cp_scores = np.load('vdn_op/score_mean.npy')
vdn_op_inter_scores = np.hstack([total_cp_scores[:,0:10],total_cp_scores[:,20:]]) .flatten()

vdn_sbrt_intra_scores = np.load('t3r02_vdn_interplay/score_mean.npy').flatten()
vdn_sbrt_intra_sim = np.load('t3r02_vdn_interplay/distance_b2a.npy').flatten()
vdn_sbrt_intra_scores = np.delete(vdn_sbrt_intra_scores, vdn_sbrt_intra_sim>0.95)
total_cp_scores = np.load('t3r02_vdn/score_mean.npy')
vdn_sbrt_inter_scores = np.hstack([total_cp_scores[:,0:10],total_cp_scores[:,20:]]).flatten()

print('vdn intra scores,', '%.2f'%(vdn_intra_scores.mean()),"+",'%.2f'%(vdn_intra_scores.std()/np.sqrt(0.5*vdn_intra_scores.shape[0])))
print('vdn inter scores,', '%.2f'%(vdn_inter_scores.mean()),"+",'%.2f'%(vdn_inter_scores.std()/np.sqrt(0.5*vdn_inter_scores.shape[0])))
print('vdn-op intra scores,', '%.2f'%(vdn_op_intra_scores.mean()),"+",'%.2f'%(vdn_op_intra_scores.std()/np.sqrt(0.5*vdn_op_intra_scores.shape[0])))
print('vdn-op inter scores,', '%.2f'%(vdn_op_inter_scores.mean()),"+",'%.2f'%(vdn_op_inter_scores.std()/np.sqrt(0.5*vdn_op_inter_scores.shape[0])))
print('vdn-sbrt intra scores,', '%.2f'%(vdn_sbrt_intra_scores.mean()),"+",'%.2f'%(vdn_sbrt_intra_scores.std()/np.sqrt(0.5*vdn_sbrt_intra_scores.shape[0])))
print('vdn-sbrt inter scores,', '%.2f'%(vdn_sbrt_inter_scores.mean()),"+",'%.2f'%(vdn_sbrt_inter_scores.std()/np.sqrt(0.5*vdn_sbrt_inter_scores.shape[0])))



