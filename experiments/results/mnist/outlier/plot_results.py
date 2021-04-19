import torch
import pickle as pkl
from models.image_recognition import MnistClassifier
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

CV = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rc('text', usetex=True)
params = {'text.latex.preamble': r'\usepackage{amsmath}'}
plt.rcParams.update(params)
test_size = 200
metrics = np.zeros((2, test_size, CV + 1))
n_inspected = [n for n in range(test_size)]

for cv in range(CV + 1):
    with open(f'simplex_cv{cv}.pkl', 'rb') as f:
        corpus = pkl.load(f)
    latents_true = corpus.test_latent_reps.to(device)
    latents_approx = corpus.latent_approx().to(device)
    residuals = ((latents_true - latents_approx) ** 2).mean(dim=-1)
    counts_simplex = []
    counts_random = []
    random_perm = torch.randperm(test_size)
    for k in range(residuals.shape[0]):
        _, top_id = torch.topk(residuals, k)
        random_id = random_perm[:k]
        count_simplex = torch.count_nonzero(top_id > 99).item()
        count_random = torch.count_nonzero(random_id > 99).item()
        counts_simplex.append(count_simplex)
        counts_random.append(count_random)
    metrics[0, :, cv] = counts_simplex
    metrics[1, :, cv] = counts_random

counts_ideal = [n if n < int(test_size/2) else int(test_size/2) for n in range(test_size)]
sns.set()
plt.plot(n_inspected, metrics[0].mean(axis=-1), label='Simplex')
plt.fill_between(n_inspected, metrics[0].mean(axis=-1) - metrics[0].std(axis=-1),
                 metrics[0].mean(axis=-1) + metrics[0].std(axis=-1), alpha=0.3)
plt.plot(n_inspected, metrics[1].mean(axis=-1), color='red', label='Random')
plt.fill_between(n_inspected, metrics[1].mean(axis=-1) - metrics[1].std(axis=-1),
                 metrics[1].mean(axis=-1) + metrics[1].std(axis=-1), alpha=0.3)
plt.plot(n_inspected, counts_ideal, color='green', label='Ideal')
plt.xlabel('Number of inspected examples')
plt.ylabel('Number of outliers detected')
plt.legend()
plt.show()
