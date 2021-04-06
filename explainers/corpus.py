import numpy as np
import cvxpy as cp
import torch
import torch.nn.functional as F


class Corpus:
    def __init__(self, examples: np.ndarray, latent_reps: np.ndarray):
        self.examples = examples
        self.latent_reps = latent_reps
        self.corpus_size = examples.shape[0]
        self.dim_latent = latent_reps.shape[-1]
        self.weights = None
        self.n_test = None

    def fit(self, test_latent_reps: np.ndarray, decimals=None):
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

    def fit_torch(self, test_latent_reps: torch.Tensor,
                  decimals=None, learning_rate=0.1, momentum=1, n_epoch=100, reg_factor=0.1):
        latent_reps = torch.from_numpy(self.latent_reps).to(test_latent_reps.device)
        n_test = test_latent_reps.shape[0]
        preweights = torch.randn((n_test, self.corpus_size), device=test_latent_reps.device, requires_grad=True)
        optimizer = torch.optim.SGD([preweights], lr=learning_rate, momentum=momentum)
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            weights = F.softmax(preweights, dim=-1)
            corpus_latent_reps = torch.einsum('ij,jk->ik', weights, latent_reps)
            error = ((corpus_latent_reps - test_latent_reps)**2).mean()
            regulator = reg_factor*(torch.abs(weights)).mean()
            loss = error + regulator
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print(f'Weight Fitting Epoch: {epoch}/{n_epoch} ; Loss: {loss.item()}')
        self.weights = torch.softmax(preweights, dim=-1).clone().detach().cpu().numpy()
        self.n_test = n_test
        return self.weights


    def decompose(self, test_id):
        assert test_id < self.n_test
        weights = self.weights[test_id]
        sort_id = np.argsort(weights)[::-1]
        return [(weights[i], self.examples[i]) for i in sort_id]


