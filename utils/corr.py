import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sensitivities = np.load('../test_data/lstmtestglove50.9.model_sensitivities.npy')
cor = np.corrcoef(sensitivities,rowvar=0)
tick = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
cmap = sns.diverging_palette(260, 10, as_cmap=True)
sns.heatmap(cor, cmap=cmap, xticklabels=tick, yticklabels=tick, square=True, cbar=False)
plt.savefig('../test_data/lstmtestglove50.9.model_sensitivities_correlation.png', bbox_inches = 'tight')

plt.show()

# sensitivities = np.load('../model/CONLL003/lstmtestglove50.9.model_sensitivities.npy')

# sens_abs = np.abs(sensitivities)
# sum_sens = np.sum(sens_abs, axis=1)
sum_sens = np.sum(sensitivities, axis=1)
index = sum_sens.argsort()[::-1][0:10]

print(sorted(index))

plt.figure(figsize=(10, 5))
x_tick = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
cmap = sns.diverging_palette(260, 10, as_cmap=True)
ax = sns.heatmap(sensitivities[sorted(index)], xticklabels=x_tick, yticklabels=sorted(index), annot=True, fmt=".2g", cmap=cmap)
ax.xaxis.set_ticks_position('top')
# plt.title(title, fontsize=18)
plt.xticks(rotation=360)
plt.yticks(rotation=360)
plt.show()
ax.figure.savefig("../test_data/top_ten_sum_sensitivities_heatmap.png")


# weight-based correlation is done in /utils/weight_similarity.py

