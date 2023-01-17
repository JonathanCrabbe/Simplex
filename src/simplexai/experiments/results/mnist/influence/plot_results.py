import argparse
import json
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser()
parser.add_argument(
    "-cv_list",
    nargs="+",
    default=[0, 1, 2, 3, 4],
    help="The list of experiment cv identifiers to plot",
    type=int,
)
parser.add_argument(
    "-k_list",
    nargs="+",
    default=[2, 5, 10, 50],
    help="The list of active corpus members considered",
    type=int,
)
args = parser.parse_args()
cv_list = args.cv_list
n_keep_list = args.k_list
r2_array = np.zeros((2, len(n_keep_list), len(cv_list)))
current_path = Path.cwd()
load_path = current_path / "experiments/results/mnist/influence"

for cv in cv_list:
    with open(load_path / f"corpus_latent_reps_cv{cv}.pkl", "rb") as f:
        corpus_latent_reps = pkl.load(f)
    with open(load_path / f"test_latent_reps_cv{cv}.pkl", "rb") as f:
        test_latent_reps = pkl.load(f)
    with open(load_path / f"influence_functions_cv{cv}.json") as f:
        influence_dic = json.load(f)

    for n, n_keep in enumerate(n_keep_list):
        with open(load_path / f"simplex_weights_cv{cv}_n{n_keep}.pkl", "rb") as f:
            weights = pkl.load(f)
        simplex_latent_reps = weights @ corpus_latent_reps
        simplex_r2 = r2_score(test_latent_reps, simplex_latent_reps)
        r2_array[0, n, cv] = simplex_r2

        if_latent_reps = np.zeros(test_latent_reps.shape)
        for example_number in range(len(test_latent_reps)):
            idx_best = influence_dic[str(example_number)]["helpful"][:n_keep]
            influence_list = [
                influence_dic[str(example_number)]["influence"][idx] for idx in idx_best
            ]
            weights = np.array(influence_list)
            weights /= weights.sum()
            latent_approx = weights @ corpus_latent_reps[idx_best, :]
            if_latent_reps[example_number, :] = latent_approx
        if_r2 = r2_score(test_latent_reps, if_latent_reps)
        r2_array[1, n, cv] = if_r2


sns.set()
sns.set_palette("colorblind")
sns.set_style("white")
plt.plot(n_keep_list, np.mean(r2_array[0], axis=-1), label="SimplEx")
plt.fill_between(
    n_keep_list,
    np.mean(r2_array[0], axis=-1) - np.std(r2_array[0], axis=-1),
    np.mean(r2_array[0], axis=-1) + np.std(r2_array[0], axis=-1),
    alpha=0.2,
)
plt.plot(n_keep_list, np.mean(r2_array[1], axis=-1), label="Influence Functions")
plt.fill_between(
    n_keep_list,
    np.mean(r2_array[1], axis=-1) - np.std(r2_array[1], axis=-1),
    np.mean(r2_array[1], axis=-1) + np.std(r2_array[1], axis=-1),
    alpha=0.2,
)
plt.xlabel(r"$K$")
plt.ylabel(r"$R^2_{\mathcal{H}}$")
plt.legend()
plt.tight_layout
plt.savefig(load_path / "r2_latent.pdf")
