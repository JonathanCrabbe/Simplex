import numpy as np
import cvxpy as cp


class Corpus:
    def __init__(self, examples: np.ndarray, latent_reps: np.ndarray):
        self.examples = examples
        self.latent_reps = latent_reps
        self.corpus_size = examples.shape[0]
        self.dim_latent = latent_reps.shape[-1]

    def explain(self, test_latent_reps: np.ndarray):
        n_examples = test_latent_reps.shape[0]
        weights = cp.Variable((n_examples, self.corpus_size))
        cost = cp.sum_squares(weights @ self.latent_reps - test_latent_reps)
        obj = cp.Minimize(cost)
        constr1 = weights @ np.ones(self.corpus_size) == np.ones(n_examples)
        constr2 = weights >= 0
        constr = [constr1, constr2]
        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(verbose=False)
        return weights.value
