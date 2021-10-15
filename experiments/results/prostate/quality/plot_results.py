import torch
import pickle as pkl
from models.tabular_data import MortalityPredictor
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import os

CV = 3
n_keep_list = [2, 5, 10, 50]
explainer_names = ['simplex', 'nn_uniform', 'nn_dist']
names_dict = {'simplex': 'SimplEx', 'nn_uniform': 'KNN Uniform', 'nn_dist': 'KNN Distance'}
line_styles = {'simplex': '-', 'nn_uniform': '--', 'nn_dist': ':'}
metric_names = ['r2_latent', 'r2_output', 'residual_latent', 'residual_output']
results_df = pd.DataFrame(columns=['explainer', 'n_keep', 'cv', 'r2_latent', 'r2_output',
                                   'residual_latent', 'residual_output'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rc('text', usetex=True)
params = {'text.latex.preamble' : r'\usepackage{amsmath}'}
plt.rcParams.update(params)
representer_metrics = np.zeros((2, CV+1))
load_path = os.path.join("experiments", "results", "prostate", "quality")

for cv in range(CV + 1):
    classifier = MortalityPredictor(n_cont=3)
    classifier.load_state_dict(torch.load(os.path.join(load_path, f'model_cv{cv}.pth')))
    classifier.to(device)
    classifier.eval()
    for n_keep in n_keep_list:
        for explainer_name in explainer_names:
            with open(os.path.join(load_path, f'{explainer_name}_cv{cv}_n{n_keep}.pkl'), 'rb') as f:
                explainer = pkl.load(f)
            explainer.to(device)
            latent_rep_approx = explainer.latent_approx()
            latent_rep_true = explainer.test_latent_reps
            output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
            output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
            latent_r2_score = sklearn.metrics.r2_score(latent_rep_true.cpu().numpy(), latent_rep_approx.cpu().numpy())
            output_r2_score = sklearn.metrics.r2_score(output_true.cpu().numpy(), output_approx.cpu().numpy())
            residual_latent = torch.sqrt(((latent_rep_true - latent_rep_approx) ** 2).mean()).item()
            residual_output = torch.sqrt(((output_true - output_approx) ** 2).mean()).item()
            results_df = results_df.append({'explainer': explainer_name, 'n_keep': n_keep, 'cv': cv,
                                            'r2_latent': latent_r2_score, 'r2_output': output_r2_score,
                                            'residual_latent': residual_latent, 'residual_output': residual_output},
                                           ignore_index=True)
    with open(os.path.join(load_path, f'representer_cv{cv}.pkl'), 'rb') as f:
        representer = pkl.load(f)
    representer.to(device)
    latent_rep_true = representer.test_latent_reps
    output_approx = representer.output_approx()
    output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
    representer_metrics[0, cv] = sklearn.metrics.r2_score(output_true.cpu().numpy(), output_approx.cpu().numpy())
    representer_metrics[1, cv] = torch.sqrt(((output_true - output_approx) ** 2).mean() / (output_true ** 2).mean()).item()

sns.set(font_scale=1.5)
sns.set_style("white")
mean_df = results_df.groupby(['explainer', 'n_keep']).aggregate('mean').unstack(level=0)
std_df = results_df.groupby(['explainer', 'n_keep']).aggregate('std').unstack(level=0)
min_df = results_df.groupby(['explainer', 'n_keep']).aggregate('min').unstack(level=0)
max_df = results_df.groupby(['explainer', 'n_keep']).aggregate('max').unstack(level=0)
q1_df = results_df.groupby(['explainer', 'n_keep']).quantile(0.25).unstack(level=0)
q3_df = results_df.groupby(['explainer', 'n_keep']).quantile(0.75).unstack(level=0)

for m, metric_name in enumerate(metric_names):
    plt.figure(m + 1)
    for explainer_name in explainer_names:
        plt.plot(n_keep_list, mean_df[metric_name, explainer_name], line_styles[explainer_name],
                 label=names_dict[explainer_name])
        '''
        plt.fill_between(n_keep_list, mean_df[metric_name, explainer_name] - std_df[metric_name, explainer_name],
                         mean_df[metric_name, explainer_name] + std_df[metric_name, explainer_name], alpha=0.2)
        plt.fill_between(n_keep_list, min_df[metric_name, explainer_name],
                         max_df[metric_name, explainer_name], alpha=0.2)    
        plt.fill_between(n_keep_list, q1_df[metric_name, explainer_name],
                         q3_df[metric_name, explainer_name], alpha=0.2)                                                       
        '''
        plt.fill_between(n_keep_list, mean_df[metric_name, explainer_name] - std_df[metric_name, explainer_name],
                         mean_df[metric_name, explainer_name] + std_df[metric_name, explainer_name], alpha=0.2)

plt.figure(1)
plt.xlabel(r'$K$')
plt.ylabel(r'$R^2_{\mathcal{H}}$')
plt.ylim(top=1.0)
plt.legend()
plt.savefig(os.path.join(load_path, 'r2_latent.pdf'), bbox_inches='tight')
plt.figure(2)
plt.xlabel(r'$K$')
plt.ylabel(r'$R^2_{\mathcal{Y}}$')
plt.ylim(top=1.0)
plt.legend()
plt.savefig(os.path.join(load_path, 'r2_output.pdf'), bbox_inches='tight')
plt.figure(3)
plt.xlabel(r'$K$')
plt.ylabel(r'$\| \hat{\boldsymbol{h}} - \boldsymbol{h} \| $')
plt.legend()
plt.savefig(os.path.join(load_path, 'residual_latent.pdf'), bbox_inches='tight')
plt.figure(4)
plt.xlabel(r'$K$')
plt.ylabel(r'$\| \hat{\boldsymbol{y}} - \boldsymbol{y} \| $')
plt.legend()
plt.savefig(os.path.join(load_path, 'residual_output.pdf'), bbox_inches='tight')

print(f'Representer metrics: r2_output = {representer_metrics[0].mean():.2g} +/- {representer_metrics[0].std():.2g}'
      f' ; residual_output = {representer_metrics[1].mean():.2g} +/- {representer_metrics[1].std():.2g}')






'''
for explainer_name, group in results_df.groupby('explainer'):
    mean_aggregation = {'r2_latent': 'mean', 'r2_output': 'mean',
                        'residual_latent': 'mean', 'residual_output': 'mean'}
    print(results_df.groupby('n_keep').aggregate('mean'))
    plt.plot(group['n_keep'], group['r2_latent'], label=f'{explainer_name} (latent)', linestyle='-')
    plt.plot(group['n_keep'], group['r2_output'], label=f'{explainer_name} (output)', linestyle=':')
    plt.ylabel('R2')
    plt.xlabel('Number of corpus samples')
    plt.legend()
plt.show()
'''