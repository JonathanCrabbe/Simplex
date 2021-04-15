import torch
import torchvision
import torch.optim as optim
import os
import sklearn
import pickle as pkl
import torch.nn.functional as F
from models.image_recognition import MnistClassifier
from explainers.corpus import Corpus
from explainers.nearest_neighbours import NearNeighLatent
from explainers.representer import Representer
from utils.schedulers import ExponentialScheduler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CV = 1
n_keep_list = [n for n in range(2, 50)]
explainer_names = ['simplex', 'nn_uniform', 'nn_dist']
results_df = pd.DataFrame(columns=['explainer', 'n_keep', 'cv', 'r2_latent', 'r2_output',
                                   'residual_latent', 'residual_output'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for cv in range(CV + 1):
    classifier = MnistClassifier()
    classifier.load_state_dict(torch.load(f'model_cv{cv}.pth'))
    classifier.to(device)
    classifier.eval()
    for n_keep in n_keep_list:
        for explainer_name in explainer_names:
            with open(f'{explainer_name}_cv{cv}_n{n_keep}.pkl', 'rb') as f:
                explainer = pkl.load(f)
            latent_rep_approx = explainer.latent_approx()
            latent_rep_true = explainer.test_latent_reps
            output_approx = classifier.latent_to_presoftmax(latent_rep_approx).detach()
            output_true = classifier.latent_to_presoftmax(latent_rep_true).detach()
            latent_r2_score = sklearn.metrics.r2_score(latent_rep_true.cpu().numpy(), latent_rep_approx.cpu().numpy())
            output_r2_score = sklearn.metrics.r2_score(output_true.cpu().numpy(), output_approx.cpu().numpy())
            residual_latent = torch.sqrt(
                ((latent_rep_true - latent_rep_approx) ** 2).mean() / (latent_rep_true ** 2).mean()).item()
            residual_output = torch.sqrt(((output_true - output_approx) ** 2).mean() / (output_true ** 2).mean()).item()
            results_df = results_df.append({'explainer': explainer_name, 'n_keep': n_keep, 'cv': cv,
                                            'r2_latent': latent_r2_score, 'r2_output': output_r2_score,
                                            'residual_latent': residual_latent, 'residual_output': residual_output},
                                           ignore_index=True)
sns.set()
for explainer_name, group in results_df.groupby('explainer'):
    plt.plot(group['n_keep'], group['r2_latent'], label=f'{explainer_name}', linestyle='-')
    plt.plot(group['n_keep'], group['r2_output'], label=f'{explainer_name}', linestyle=':')
    plt.legend()
plt.show()
