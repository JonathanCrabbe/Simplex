import torch


class Representer:

    def __init__(self, corpus_latent_reps: torch.Tensor, corpus_probas: torch.Tensor, corpus_true_classes: torch.Tensor,
                 reg_factor: torch.Tensor):
        self.corpus_latent_reps = corpus_latent_reps
        self.corpus_probas = corpus_probas
        self.corpus_true_classes = corpus_true_classes
        self.reg_factor = reg_factor
        self.corpus_size, self.dim_latent = corpus_latent_reps.shape
        self.num_classes = corpus_probas.shape[-1]
        self.test_latent_reps = None
        self.test_size = None

    def fit(self, test_latent_reps, n_keep: int = 5):
        self.test_latent_reps = test_latent_reps
        self.test_size = test_latent_reps.shape[0]
        projections = torch.einsum('ij kj -> ik', test_latent_reps, self.corpus_latent_reps)
        projections = projections.reshape(self.test_size, self.corpus_size, 1)
        alpha = (- self.corpus_probas + self.corpus_true_classes)/(2*self.reg_factor*n_keep)
        alpha = alpha.reshape(1, self.corpus_size, self.num_classes)
        weights = alpha * projections
        # Q: how to extract the n_keep most relevant examples from the corpus?

