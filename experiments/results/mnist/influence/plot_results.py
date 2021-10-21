import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import r2_score

CV = 3
n_keep_list = [2, 3, 4, 5, 10, 20, 30, 40, 50]
r2_array = np.zeros((2, len(n_keep_list), CV+1))

for cv in range(CV+1):
    with open(f'corpus_latent_reps_cv{cv}.pkl', 'rb') as f:
        corpus_latent_reps = pkl.load(f)
    with open(f'test_latent_reps_cv{cv}.pkl', 'rb') as f:
        test_latent_reps = pkl.load(f)
    with open(f'influence_functions_cv{cv}.json') as f:
        influence_dic = json.load(f)

    for n, n_keep in enumerate(n_keep_list):
        with open(f'simplex_weights_cv{cv}_n{n_keep}.pkl', 'rb') as f:
            weights = pkl.load(f)
        simplex_latent_reps = weights @ corpus_latent_reps
        simplex_r2 = r2_score(test_latent_reps, simplex_latent_reps)
        r2_array[0, n, cv] = simplex_r2

        if_latent_reps = np.zeros(test_latent_reps.shape)
        for example_number in range(len(test_latent_reps)):
            idx_best = influence_dic[str(example_number)]['helpful'][:n_keep]
            influence_list = [influence_dic[str(example_number)]['influence'][idx] for idx in idx_best]
            weights = np.array(influence_list)
            weights /= weights.sum()
            latent_approx = weights @ corpus_latent_reps[idx_best, :]
            if_latent_reps[example_number, :] = latent_approx
        if_r2 = r2_score(test_latent_reps, if_latent_reps)
        r2_array[1, n, cv] = if_r2

for k in [0, 3, 4, 5, -1]:
    simplex_avg, simplex_std = r2_array[0, k, :].mean(), r2_array[0, k, :].std()
    if_avg, if_std = r2_array[1, k, :].mean(), r2_array[1, k, :].std()
    print(f'For K = {n_keep_list[k]}: SimplEx R2 = {simplex_avg:.2g} +/- {simplex_std:.2g} ; '
          f'IF R2 = {if_avg:.2g} +/- {if_std:.2g}.')

sns.set(font_scale=1.5)
sns.set_style("white")
plt.plot(n_keep_list, np.mean(r2_array[0], axis=-1))
plt.fill_between(n_keep_list, np.mean(r2_array[0], axis=-1) - np.std(r2_array[0], axis=-1),
                 np.mean(r2_array[0], axis=-1) + np.std(r2_array[0], axis=-1),  alpha=0.2)
plt.plot(n_keep_list, np.mean(r2_array[1], axis=-1))
plt.fill_between(n_keep_list, np.mean(r2_array[1], axis=-1) - np.std(r2_array[1], axis=-1),
                 np.mean(r2_array[1], axis=-1) + np.std(r2_array[1], axis=-1),  alpha=0.2)
plt.xlabel(r'$K$')
plt.ylabel(r'$R^2_{\mathcal{H}}$')
plt.show()






