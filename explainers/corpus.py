import numpy as np
import cvxpy as cp
import pandas as pd
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt


class Corpus:
    def __init__(self, corpus_examples: torch.Tensor, corpus_latent_reps: torch.Tensor):
        self.corpus_examples = corpus_examples
        self.corpus_latent_reps = corpus_latent_reps
        self.corpus_size = corpus_examples.shape[0]
        self.dim_latent = corpus_latent_reps.shape[-1]
        self.weights = None
        self.n_test = None
        self.hist = None
        self.test_examples = None
        self.test_latent_reps = None

    def fit_cvxpy(self, test_latent_reps: np.ndarray, decimals=None):
        n_test = test_latent_reps.shape[0]
        weights = cp.Variable((n_test, self.corpus_size))
        cost = cp.sum_squares(weights @ self.corpus_latent_reps - test_latent_reps)
        obj = cp.Minimize(cost)
        constr1 = weights @ np.ones(self.corpus_size) == np.ones(n_test)
        constr2 = weights >= 0
        constr = [constr1, constr2]
        prob = cp.Problem(obj, constr)
        prob.solve(verbose=False)
        weights = weights.value
        if decimals:
            weights = np.round(weights, decimals)
        self.weights = weights
        self.n_test = n_test
        return weights

    def fit(self, test_examples: torch.Tensor, test_latent_reps: torch.Tensor,
            learning_rate=1.0, momentum=0.5, n_epoch=10000, reg_factor=1.0, fraction_keep=0.01,
            reg_factor_scheduler=None):
        n_test = test_latent_reps.shape[0]
        preweights = torch.zeros((n_test, self.corpus_size), device=test_latent_reps.device, requires_grad=True)
        optimizer = torch.optim.SGD([preweights], lr=learning_rate, momentum=momentum)
        hist = np.zeros((0, 2))
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            weights = F.softmax(preweights, dim=-1)
            corpus_latent_reps = torch.einsum('ij,jk->ik', weights, self.corpus_latent_reps)
            error = ((corpus_latent_reps - test_latent_reps) ** 2).mean()
            weights_sorted = torch.sort(weights)[0]
            regulator = (weights_sorted[:, :int((1 - fraction_keep) * self.corpus_size)]).mean()
            loss = error + reg_factor * regulator
            loss.backward()
            optimizer.step()
            if epoch % (n_epoch / 10) == 0:
                print(f'Weight Fitting Epoch: {epoch}/{n_epoch} ; Error: {error.item():.3g} ;'
                      f' Regulator: {regulator.item():.3g}')
            if reg_factor_scheduler:
                reg_factor = reg_factor_scheduler.step(reg_factor)
            hist = np.concatenate((hist,
                                   np.array([error.item(), regulator.item()]).reshape(1, 2)),
                                  axis=0)
        self.weights = torch.softmax(preweights, dim=-1).clone().detach().cpu().numpy()
        self.test_examples = test_examples
        self.test_latent_reps = test_latent_reps
        self.n_test = n_test
        self.hist = hist
        return self.weights

    def decompose(self, test_id):
        assert test_id < self.n_test
        weights = self.weights[test_id]
        sort_id = np.argsort(weights)[::-1]
        return [(weights[i], self.corpus_examples[i]) for i in sort_id]

    def plot_hist(self):
        sns.set()
        fig, axs = plt.subplots(2, sharex=True)
        epochs = [e for e in range(self.hist.shape[0])]
        axs[0].plot(epochs, self.hist[:, 0])
        axs[0].set(ylabel='Error')
        axs[1].plot(epochs, self.hist[:, 1])
        axs[1].set(xlabel='Epoch', ylabel='Regulator')
        plt.show()

    def residual(self, test_id: int, normalize: bool = True):
        latent_rep = self.test_latent_reps[test_id].clone().detach().cpu().numpy()
        corpus_latent_rep = (self.weights[test_id]) @ (self.corpus_latent_reps.clone().detach().cpu().numpy())
        residual = np.sum((latent_rep - corpus_latent_rep) ** 2)
        if normalize:
            residual /= np.sum(latent_rep**2)
        return np.sqrt(residual)

    def plot_residuals_CDF(self):
        sns.set()
        df = pd.DataFrame({'residuals': [self.residual(test_id) for test_id in range(self.n_test)]})
        stats_df = df \
            .groupby('residuals') \
            ['residuals'] \
            .agg('count') \
            .pipe(pd.DataFrame) \
            .rename(columns={'residuals': 'frequency'})
        stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
        stats_df['cdf'] = stats_df['pdf'].cumsum()
        stats_df = stats_df.reset_index()
        stats_df.plot(x='residuals', y='cdf', grid=True,
                      xlabel='Normalized Residual', ylabel='Cumulative Distribution', legend=False)

    def to(self, device: torch.device):
        self.corpus_examples = self.corpus_examples.to(device)
        self.corpus_latent_reps = self.corpus_latent_reps.to(device)
        self.test_examples = self.test_examples.to(device)
        self.test_latent_reps = self.test_latent_reps.to(device)
