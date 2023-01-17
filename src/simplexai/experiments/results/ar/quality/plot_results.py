import argparse
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch

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
k_list = args.k_list
explainer_names = ["simplex", "knn_uniform", "knn_dist"]
names_dict = {
    "simplex": "SimplEx",
    "knn_uniform": "KNN Uniform",
    "knn_dist": "KNN Distance",
}
line_styles = {"simplex": "-", "knn_uniform": "--", "knn_dist": ":"}
metric_names = ["r2_latent", "r2_output", "residual_latent", "residual_output"]
results_df = pd.DataFrame(
    columns=[
        "explainer",
        "n_keep",
        "cv",
        "r2_latent",
        "r2_output",
        "residual_latent",
        "residual_output",
    ]
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rc("text", usetex=True)
params = {"text.latex.preamble": r"\usepackage{amsmath}"}
plt.rcParams.update(params)
load_dir = Path.cwd() / "experiments/results/ar/quality/"

for cv in cv_list:
    for k in k_list:
        with open(load_dir / f"true_k{k}_cv{cv}.pkl", "rb") as f:
            latent_true, output_true = pkl.load(f)
        for explainer_name in explainer_names:
            with open(load_dir / f"{explainer_name}_k{k}_cv{cv}.pkl", "rb") as f:
                latent_approx, output_approx = pkl.load(f)

            latent_r2 = sklearn.metrics.r2_score(latent_true, latent_approx)
            output_r2 = sklearn.metrics.r2_score(output_true, output_approx)
            latent_residual = np.sqrt(
                ((latent_true - latent_approx) ** 2).mean()
            ).item()
            output_residual = np.sqrt(
                ((output_true - output_approx) ** 2).mean()
            ).item()
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame.from_dict(
                        {
                            "explainer": [explainer_name],
                            "k": [k],
                            "cv": [cv],
                            "r2_latent": [latent_r2],
                            "r2_output": [output_r2],
                            "residual_latent": [latent_residual],
                            "residual_output": [output_residual],
                        }
                    ),
                ],
                ignore_index=True,
            )

sns.set()
sns.set_style("white")
sns.set_palette("colorblind")
mean_df = results_df.groupby(["explainer", "k"]).aggregate("mean").unstack(level=0)
std_df = results_df.groupby(["explainer", "k"]).aggregate("std").unstack(level=0)
min_df = results_df.groupby(["explainer", "k"]).aggregate("min").unstack(level=0)
max_df = results_df.groupby(["explainer", "k"]).aggregate("max").unstack(level=0)
q1_df = results_df.groupby(["explainer", "k"]).quantile(0.25).unstack(level=0)
q3_df = results_df.groupby(["explainer", "k"]).quantile(0.75).unstack(level=0)

for m, metric_name in enumerate(metric_names):
    plt.figure(m + 1)
    for explainer_name in explainer_names:
        plt.plot(
            k_list,
            mean_df[metric_name, explainer_name],
            line_styles[explainer_name],
            label=names_dict[explainer_name],
        )
        plt.fill_between(
            k_list,
            mean_df[metric_name, explainer_name] - std_df[metric_name, explainer_name],
            mean_df[metric_name, explainer_name] + std_df[metric_name, explainer_name],
            alpha=0.2,
        )

plt.figure(1)
plt.xlabel(r"$K$")
plt.ylabel(r"$R^2_{\mathcal{H}}$")
plt.ylim(top=1.0)
plt.legend()
plt.savefig(load_dir / "r2_latent.pdf", bbox_inches="tight")
plt.figure(2)
plt.xlabel(r"$K$")
plt.ylabel(r"$R^2_{\mathcal{Y}}$")
plt.ylim(top=1.0)
plt.legend()
plt.savefig(load_dir / "r2_output.pdf", bbox_inches="tight")
plt.figure(3)
plt.xlabel(r"$K$")
plt.ylabel(r"$\| \hat{\boldsymbol{h}} - \boldsymbol{h} \|  $")
plt.legend()
plt.savefig(load_dir / "residual_latent.pdf", bbox_inches="tight")
plt.figure(4)
plt.xlabel(r"$K$")
plt.ylabel(r"$\| \hat{\boldsymbol{y}} - \boldsymbol{y} \| $")
plt.legend()
plt.savefig(load_dir / "residual_output.pdf", bbox_inches="tight")
