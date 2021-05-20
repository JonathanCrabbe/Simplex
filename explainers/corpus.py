import numpy as np
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
        self.jacobian_projections = None

    def fit(self, test_examples: torch.Tensor, test_latent_reps: torch.Tensor,
            learning_rate=1.0, momentum=0.5, n_epoch=10000, reg_factor=1.0, n_keep: int = 5,
            reg_factor_scheduler=None):
        n_test = test_latent_reps.shape[0]
        preweights = torch.zeros((n_test, self.corpus_size), device=test_latent_reps.device, requires_grad=True)
        # optimizer = torch.optim.SGD([preweights], lr=learning_rate, momentum=momentum)
        optimizer = torch.optim.Adam([preweights])
        hist = np.zeros((0, 2))
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            weights = F.softmax(preweights, dim=-1)
            corpus_latent_reps = torch.einsum('ij,jk->ik', weights, self.corpus_latent_reps)
            error = ((corpus_latent_reps - test_latent_reps) ** 2).mean()
            weights_sorted = torch.sort(weights)[0]
            regulator = (weights_sorted[:, : (self.corpus_size - n_keep)]).mean()
            loss = error + reg_factor * regulator
            loss.backward()
            optimizer.step()
            if epoch % (n_epoch / 5) == 0:
                print(f'Weight Fitting Epoch: {epoch}/{n_epoch} ; Error: {error.item():.3g} ;'
                      f' Regulator: {regulator.item():.3g}')
            if reg_factor_scheduler:
                reg_factor = reg_factor_scheduler.step(reg_factor)
            hist = np.concatenate((hist,
                                   np.array([error.item(), regulator.item()]).reshape(1, 2)),
                                  axis=0)
        self.weights = torch.softmax(preweights, dim=-1).detach()
        self.test_examples = test_examples
        self.test_latent_reps = test_latent_reps
        self.n_test = n_test
        self.hist = hist

    def to(self, device: torch.device):
        self.corpus_examples = self.corpus_examples.to(device)
        self.corpus_latent_reps = self.corpus_latent_reps.to(device)
        self.test_examples = self.test_examples.to(device)
        self.test_latent_reps = self.test_latent_reps.to(device)
        self.weights = self.weights.to(device)


    def latent_approx(self):
        approx_reps = self.weights @ self.corpus_latent_reps
        return approx_reps

    def decompose(self, test_id, return_id = False):
        assert test_id < self.n_test
        weights = self.weights[test_id].cpu().numpy()
        sort_id = np.argsort(weights)[::-1]
        if return_id:
            return [(weights[i], self.corpus_examples[i], self.jacobian_projections[i]) for i in sort_id], sort_id
        else:
            return [(weights[i], self.corpus_examples[i], self.jacobian_projections[i]) for i in sort_id]

    def plot_hist(self):
        sns.set()
        fig, axs = plt.subplots(2, sharex=True)
        epochs = [e for e in range(self.hist.shape[0])]
        axs[0].plot(epochs, self.hist[:, 0])
        axs[0].set(ylabel='Error')
        axs[1].plot(epochs, self.hist[:, 1])
        axs[1].set(xlabel='Epoch', ylabel='Regulator')
        plt.show()

    def jacobian_projection(self, test_id, model: torch.nn.Module, input_baseline, n_bins=100):
        corpus_inputs = self.corpus_examples.clone().requires_grad_()
        input_shift = self.corpus_examples - input_baseline
        #latent_shift = self.test_latent_reps[test_id:test_id+1] - model.latent_representation(input_baseline)
        latent_shift = self.latent_approx()[test_id:test_id + 1] - model.latent_representation(input_baseline)
        latent_shift_sqrdnorm = torch.sum(latent_shift**2, dim=-1, keepdim=True)
        input_grad = torch.zeros(corpus_inputs.shape, device=corpus_inputs.device)
        for n in range(1, n_bins + 1):
            t = n / n_bins
            input = input_baseline + t * (corpus_inputs - input_baseline)
            latent_reps = model.latent_representation(input)
            latent_reps.backward(gradient=latent_shift/latent_shift_sqrdnorm)
            input_grad += corpus_inputs.grad
            corpus_inputs.grad.data.zero_()
        self.jacobian_projections = input_shift * input_grad / (n_bins)
        return self.jacobian_projections

'''

    def residual(self, test_id: int, normalize: bool = True):
        true_rep = self.test_latent_reps[test_id].clone().detach().cpu().numpy()
        corpus_approx_rep = (self.weights[test_id]) @ (self.corpus_latent_reps.clone().detach().cpu().numpy())
        residual = np.sum((true_rep - corpus_approx_rep) ** 2)
        if normalize:
            residual /= np.sum(true_rep**2)
        return np.sqrt(residual)

    def residual_statistics(self, normalize: bool = True):
        true_reps = self.test_latent_reps.clone().detach().cpu().numpy()
        corpus_approx_reps = self.weights @ (self.corpus_latent_reps.clone().detach().cpu().numpy())
        residuals = np.sum((true_reps - corpus_approx_reps) ** 2, axis=-1)
        if normalize:
            residuals /= np.sum(true_reps ** 2, axis=-1)
        residuals = np.sqrt(residuals)
        return np.mean(residuals), np.std(residuals)

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

    def latent_r2_score(self):
        true_reps = self.test_latent_reps.clone().detach().cpu().numpy()
        corpus_approx_reps = self.weights @ (self.corpus_latent_reps.clone().detach().cpu().numpy())
        return sklearn.metrics.r2_score(true_reps, corpus_approx_reps)


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
'''
