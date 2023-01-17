import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "-cv_list",
    nargs="+",
    required=True,
    default=[0, 1, 2, 3, 4],
    help="The list of experiment cv identifiers to plot",
    type=int,
)
args = parser.parse_args()
cv_list = args.cv_list
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rc("text", usetex=True)
params = {"text.latex.preamble": r"\usepackage{amsmath}"}
plt.rcParams.update(params)
test_size = 200
metrics = np.zeros((4, test_size, len(cv_list)))
n_inspected = [n for n in range(test_size)]
current_path = Path.cwd()
load_path = current_path / "experiments/results/mnist/outlier"


for cv in cv_list:
    with open(load_path / f"simplex_cv{cv}.pkl", "rb") as f:
        simplex = pkl.load(f)
    with open(load_path / f"nn_dist_cv{cv}.pkl", "rb") as f:
        nn_dist = pkl.load(f)
    with open(load_path / f"nn_uniform_cv{cv}.pkl", "rb") as f:
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

counts_ideal = [
    n if n < int(test_size / 2) else int(test_size / 2) for n in range(test_size)
]
sns.set(font_scale=1.5)
sns.set_style("white")
sns.set_palette("colorblind")
plt.plot(n_inspected, metrics[0].mean(axis=-1), label="Simplex")
plt.fill_between(
    n_inspected,
    metrics[0].mean(axis=-1) - metrics[0].std(axis=-1),
    metrics[0].mean(axis=-1) + metrics[0].std(axis=-1),
    alpha=0.3,
)
plt.plot(n_inspected, metrics[1].mean(axis=-1), label="5NN Distance")
plt.fill_between(
    n_inspected,
    metrics[1].mean(axis=-1) - metrics[1].std(axis=-1),
    metrics[1].mean(axis=-1) + metrics[1].std(axis=-1),
    alpha=0.3,
)
plt.plot(n_inspected, metrics[2].mean(axis=-1), label="5NN Uniform")
plt.fill_between(
    n_inspected,
    metrics[2].mean(axis=-1) - metrics[2].std(axis=-1),
    metrics[2].mean(axis=-1) + metrics[2].std(axis=-1),
    alpha=0.3,
)
plt.plot(n_inspected, metrics[3].mean(axis=-1), label="Random")
plt.fill_between(
    n_inspected,
    metrics[3].mean(axis=-1) - metrics[3].std(axis=-1),
    metrics[3].mean(axis=-1) + metrics[3].std(axis=-1),
    alpha=0.3,
)
plt.plot(n_inspected, counts_ideal, label="Maximal")
plt.xlabel("Number of images inspected")
plt.ylabel("Number of EMNIST detected")
plt.legend()
plt.savefig(load_path / "outlier.pdf", bbox_inches="tight")
