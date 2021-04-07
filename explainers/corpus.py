import numpy as np
import cvxpy as cp
import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt


class Corpus:
    def __init__(self, examples: np.ndarray, latent_reps: np.ndarray):
        self.examples = examples
        self.latent_reps = latent_reps
        self.corpus_size = examples.shape[0]
        self.dim_latent = latent_reps.shape[-1]
        self.weights = None
        self.n_test = None
        self.hist = None

    def fit_cvxpy(self, test_latent_reps: np.ndarray, decimals=None):
        n_test = test_latent_reps.shape[0]
        weights = cp.Variable((n_test, self.corpus_size))
        cost = cp.sum_squares(weights @ self.latent_reps - test_latent_reps)
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

    def fit(self, test_latent_reps: torch.Tensor,
            learning_rate=0.01, momentum=0.5, n_epoch=2000, reg_factor=1.0, fraction_keep=0.01,
            reg_factor_scheduler=None):
        latent_reps = torch.from_numpy(self.latent_reps).to(test_latent_reps.device)
        n_test = test_latent_reps.shape[0]
        preweights = torch.zeros((n_test, self.corpus_size), device=test_latent_reps.device, requires_grad=True)
        optimizer = torch.optim.SGD([preweights], lr=learning_rate, momentum=momentum)
        hist = np.zeros((0, 2))
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            weights = F.softmax(preweights, dim=-1)
            corpus_latent_reps = torch.einsum('ij,jk->ik', weights, latent_reps)
            error = ((corpus_latent_reps - test_latent_reps) ** 2).mean()
            weights_sorted = torch.sort(weights)[0]
            regulator = (weights_sorted[:, :int((1-fraction_keep)*self.corpus_size)]).mean()
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
        self.n_test = n_test
        self.hist = hist
        return self.weights

    def decompose(self, test_id):
        assert test_id < self.n_test
        weights = self.weights[test_id]
        sort_id = np.argsort(weights)[::-1]
        return [(weights[i], self.examples[i]) for i in sort_id]

    def plot_hist(self):
        sns.set()
        fig, axs = plt.subplots(2, sharex=True)
        epochs = [e for e in range(self.hist.shape[0])]
        axs[0].plot(epochs, self.hist[:, 0])
        axs[0].set(ylabel='Error')
        axs[1].plot(epochs, self.hist[:, 1])
        axs[1].set(xlabel='Epoch', ylabel='Regulator')
        plt.show()


