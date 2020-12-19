import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import spatial
import operator

weights = np.load('../weights.npy')
ind_to_tag = ['PAD', 'B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC']
output_tag = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
def similarity(w1, w2, combination):
    '''
    Cosine similarity is 1-distance
    :param w1: row in weight matrix of first tag
    :param w2: row in weight matrix of tag 2
    :return: cosine similarity between them
    '''
    ##print("combination = {}".format(combination))
    ##print("combination = {}\n   row 1: {}\n row 2: {}".format(combination, w1, w2))
    # 1 - dist is similarity
    sim = 1- spatial.distance.cosine(w1, w2)
    # print("similarity: {}".format(sim))
    return sim


def print_sims(sorted_sim_dict):
    for comb, sim in sorted_sim_dict.items():
#         print("{}\t{}\t{}".format(ind_to_tag[comb[0]], ind_to_tag[comb[1]], sim))
        print("{}\t{}\t{}".format(output_tag[comb[0]], output_tag[comb[1]], sim))

sim_matrix = np.zeros((9, 9))
comb = combinations([0,1,2,3,4,5,6,7,8],2)  # omit 0, the padding tag
# comb = combinations([1,2,3,4,5,6,7,8,9],2)  # omit 0, the padding tag
sim_dict = {}
sims = {}
for c in list(comb):
    s = similarity(weights[ind_to_tag.index(output_tag[c[0]])], weights[ind_to_tag.index(output_tag[c[1]])], c)
    sim_dict[c] = s
    sims[s] = c
    sim_matrix[c[0], c[1]] = s
    sim_matrix[c[1], c[0]] = s

for i in range(9):
    sim_matrix[i, i] = 1
tick = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
cmap = sns.diverging_palette(260, 10, as_cmap=True)

# weight_temp = np.zeros((9, 50))
# for i in range(len(x_tick_output)):
#     sensitivities_temp[i] = sensitivities[ind_to_tag.index(output_tag[i])]
# cor = np.corrcoef(sensitivities,rowvar=1)
    
sns.heatmap(sim_matrix, cmap=cmap, xticklabels=tick, yticklabels=tick, square=True)#, cbar=False)
plt.savefig('../../model/CONLL003/lstmtestglove50_4.10.model_weight_correlation.png', bbox_inches = 'tight')

# sort by descending similarity
# sorted_d = dict(sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True))

# Does this have a bug because getting negative numbers?
# print_sims(sorted_d)
print_sims(sim_dict)



