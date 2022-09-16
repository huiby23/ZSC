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

scores = np.load('base_4_methods_10_seeds/score_mean.npy')
similarity = np.load('base_4_methods_10_seeds/distance_b2a.npy')
model_type = ['IQL','VDN','SAD','OBL']


person_mat = np.zeros((4,5))
similarity_mat_mean = np.zeros((4,5))
similarity_mat_ste = np.zeros((4,5))
cross_play_mat_mean = np.zeros((4,5))
cross_play_mat_ste = np.zeros((4,5))



for x_idx in range(4):
    for y_idx in range(4):
        mini_scores = scores[x_idx*10:(x_idx+1)*10, y_idx*10:(y_idx+1)*10].flatten()
        mini_similarity = similarity[x_idx*10:(x_idx+1)*10, y_idx*10:(y_idx+1)*10].flatten()

        mini_scores = np.delete(mini_scores,mini_similarity>0.95)
        mini_similarity = np.delete(mini_similarity,mini_similarity>0.95) 

        person_mat[x_idx,y_idx] = pearson_coef(mini_scores, mini_similarity)
        similarity_mat_mean[x_idx,y_idx] = np.mean(mini_similarity)
        similarity_mat_ste[x_idx,y_idx] = np.std(mini_similarity)/np.sqrt(mini_similarity.shape[0])
        cross_play_mat_mean[x_idx,y_idx] = np.mean(mini_scores)
        cross_play_mat_ste[x_idx,y_idx] = np.std(mini_scores)/np.sqrt(mini_scores.shape[0])

    mid_scores = scores[x_idx*10:(x_idx+1)*10, :].flatten()
    mid_similarity = similarity[x_idx*10:(x_idx+1)*10, :].flatten()
    mid_scores = np.delete(mid_scores,mid_similarity>0.95)
    mid_similarity = np.delete(mid_similarity,mid_similarity>0.95) 

    person_mat[x_idx,4] = pearson_coef(mid_scores, mid_similarity)
    similarity_mat_mean[x_idx,4] = np.mean(mid_similarity)
    similarity_mat_ste[x_idx,4] = np.std(mid_similarity)/np.sqrt(mid_similarity.shape[0])
    cross_play_mat_mean[x_idx,4] = np.mean(mid_scores)
    cross_play_mat_ste[x_idx,4] = np.std(mid_scores)/np.sqrt(mid_scores.shape[0])

print('person_mat:',person_mat)
print('similarity_mat_mean:',similarity_mat_mean)
print('similarity_mat_ste:',similarity_mat_ste)
print('cross_play_mat_mean:',cross_play_mat_mean)
print('cross_play_mat_ste:',cross_play_mat_ste)








'''
#get intra-play scores
self_play_records = ['sad_op_selfplay','t3r02_op_selfplay']
for record in self_play_records:
    score_data = np.load(record+'/score_mean.npy').flatten()
    slr_data = np.load(record+'/distance_b2a.npy').flatten()
    print('data:', record)
    final_data = np.delete(score_data,slr_data>0.95)
    print(final_data.mean())

#get inter-play scores

aaa = np.load('t3r02_op/score_mean.npy')
print('sad+op+sbrt,',np.hstack([aaa[:,0:20],aaa[:,30:]]).mean())
bbb = np.load('sad_op/score_mean.npy')
print('sad+op,',np.hstack([bbb[:,0:20],bbb[:,30:]]).mean())
ccc = np.load('t3r02_sad/score_mean.npy')
print('sad+sbrt,',np.hstack([ccc[:,0:20],ccc[:,30:]]).mean())
ddd = np.load('t3r02_vdn/score_mean.npy')
print('vdn+sbrt,',np.hstack([ddd[:,0:10],ddd[:,20:]]).mean())
eee = np.load('t3r02_iql/score_mean.npy')
print('iql+sbrt,',eee[:,10:].mean())
'''
