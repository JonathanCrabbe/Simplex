import torch
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import seaborn as sns
import pandas as pd
import sklearn


class NearNeighLatent:
    def __init__(self, corpus_examples, corpus_latent_reps, weights_type: str = 'uniform'):
        if type(corpus_examples) == torch.Tensor:
            corpus_examples = corpus_examples.clone().detach().cpu().numpy()
        if type(corpus_latent_reps) == torch.Tensor:
            corpus_latent_reps = corpus_latent_reps.clone().detach().cpu().numpy()
        self.corpus_examples = corpus_examples
        self.corpus_latent_reps = corpus_latent_reps
        self.corpus_size = corpus_examples.shape[0]
        self.dim_latent = corpus_latent_reps.shape[-1]
        self.weights_type = weights_type
        self.weights = None
        self.n_test = None
        self.test_examples = None
        self.test_latent_reps = None
        self.regressor = None

    def fit(self, test_examples, test_latent_reps, n_keep: int = 5):
        if type(test_examples) == torch.Tensor:
            test_examples = test_examples.clone().detach().cpu().numpy()
        if type(test_latent_reps) == torch.Tensor:
            test_latent_reps = test_latent_reps.clone().detach().cpu().numpy()
        regressor = KNeighborsRegressor(n_neighbors=n_keep, weights=self.weights_type)
        regressor.fit(self.corpus_latent_reps, self.corpus_latent_reps)
        self.test_examples = test_examples
        self.n_test = test_examples.shape[0]
        self.test_latent_reps = test_latent_reps
        self.regressor = regressor

    def residual(self, test_id: int, normalize: bool = True):
        true_rep = self.test_latent_reps[test_id]
        corpus_approx_rep = self.regressor.predict([self.test_latent_reps[test_id]])
        residual = np.sum((true_rep - corpus_approx_rep) ** 2)
        if normalize:
            residual /= np.sum(true_rep**2)
        return np.sqrt(residual)

    def residual_statistics(self, normalize: bool = True):
        true_reps = self.test_latent_reps
        corpus_approx_reps = self.regressor.predict(self.test_latent_reps)
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

    def r2_score(self):
        true_reps = self.test_latent_reps
        corpus_approx_reps = self.regressor.predict(self.test_latent_reps)
        return sklearn.metrics.r2_score(true_reps, corpus_approx_reps)

