import torch
import pickle as pkl
from models.image_recognition import MnistClassifier
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

CV = 9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rc('text', usetex=True)
params = {'text.latex.preamble': r'\usepackage{amsmath}'}
plt.rcParams.update(params)
test_size = 200
metrics = np.zeros((4, test_size, CV + 1))
n_inspected = [n for n in range(test_size)]

for cv in range(CV + 1):
    with open(f'simplex_cv{cv}.pkl', 'rb') as f:
        simplex = pkl.load(f)
    with open(f'nn_dist_cv{cv}.pkl', 'rb') as f:
        nn_dist = pkl.load(f)
    with open(f'nn_uniform_cv{cv}.pkl', 'rb') as f:
        nn_uniform = pkl.load(f)
    latents_true = simplex.test_latent_reps.to(device)
    simplex_latent_approx = simplex.latent_approx().to(device)
    nn_dist_latent_approx = nn_dist.latent_approx().to(device)
    nn_uniform_latent_approx = nn_uniform.latent_approx().to(device)
    simplex_residuals = ((latents_true - simplex_latent_approx) ** 2).mean(dim=-1)
    nn_dist_residuals = ((latents_true - nn_dist_latent_approx) ** 2).mean(dim=-1)
    nn_uniform_residuals = ((latents_true - nn_uniform_latent_approx) ** 2).mean(dim=-1)
    counts_simplex = []
    counts_nn_dist = []
    counts_nn_uniform = []
    counts_random = []
    random_perm = torch.randperm(test_size)
    for k in range(simplex_residuals.shape[0]):
        _, simplex_top_id = torch.topk(simplex_residuals, k)
        _, nn_dist_top_id = torch.topk(nn_dist_residuals, k)
        _, nn_uniform_top_id = torch.topk(nn_uniform_residuals, k)
        random_id = random_perm[:k]
        count_simplex = torch.count_nonzero(simplex_top_id > 99).item()
        count_nn_dist = torch.count_nonzero(nn_dist_top_id > 99).item()
        count_nn_uniform = torch.count_nonzero(nn_uniform_top_id > 99).item()
        count_random = torch.count_nonzero(random_id > 99).item()
        counts_simplex.append(count_simplex)
        counts_nn_dist.append(count_nn_dist)
        counts_nn_uniform.append(count_nn_uniform)
        counts_random.append(count_random)
    metrics[0, :, cv] = counts_simplex
    metrics[1, :, cv] = counts_nn_dist
    metrics[2, :, cv] = counts_nn_uniform
    metrics[3, :, cv] = counts_random

counts_ideal = [n if n < int(test_size/2) else int(test_size/2) for n in range(test_size)]
sns.set(font_scale=1.5)
sns.set_style("white")
plt.plot(n_inspected, metrics[0].mean(axis=-1), label='Simplex')
plt.fill_between(n_inspected, metrics[0].mean(axis=-1) - metrics[0].std(axis=-1),
                 metrics[0].mean(axis=-1) + metrics[0].std(axis=-1), alpha=0.3)
plt.plot(n_inspected, metrics[1].mean(axis=-1), label='5NN Distance')
plt.fill_between(n_inspected, metrics[1].mean(axis=-1) - metrics[1].std(axis=-1),
                 metrics[1].mean(axis=-1) + metrics[1].std(axis=-1), alpha=0.3)
plt.plot(n_inspected, metrics[2].mean(axis=-1), label='5NN Uniform')
plt.fill_between(n_inspected, metrics[2].mean(axis=-1) - metrics[2].std(axis=-1),
                 metrics[2].mean(axis=-1) + metrics[2].std(axis=-1), alpha=0.3)
plt.plot(n_inspected, metrics[3].mean(axis=-1), label='Random')
plt.fill_between(n_inspected, metrics[3].mean(axis=-1) - metrics[3].std(axis=-1),
                 metrics[3].mean(axis=-1) + metrics[3].std(axis=-1), alpha=0.3)
plt.plot(n_inspected, counts_ideal, label='Ideal')
plt.xlabel('Number of images inspected')
plt.ylabel('Number of EMNIST detected')
plt.legend()
plt.savefig('outlier.pdf', bbox_inches='tight')
plt.show()