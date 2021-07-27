import pytorch_influence_functions as ptif
import torch

class NearNeighLatent:
    def __init__(self, corpus_examples: torch.Tensor, corpus_latent_reps: torch.Tensor, weights_type: str = 'uniform'):
        self.corpus_examples = corpus_examples
        self.corpus_latent_reps = corpus_latent_reps
        self.corpus_size = corpus_examples.shape[0]
        self.dim_latent = corpus_latent_reps.shape[-1]
        self.weights_type = weights_type
        self.n_test = None
        self.test_examples = None
        self.test_latent_reps = None
        self.regressor = None

    def fit(self, test_examples: torch.Tensor, test_latent_reps: torch.Tensor, n_keep: int = 5):
        print("coucou")


    def latent_approx(self):
        approx_reps = self.regressor.predict(self.test_latent_reps.clone().detach().cpu().numpy())
        return torch.from_numpy(approx_reps).type(torch.float32).to(self.test_latent_reps.device)

    def to(self, device:torch.device):
        self.corpus_examples = self.corpus_examples.to(device)
        self.corpus_latent_reps = self.corpus_latent_reps.to(device)
        self.test_examples = self.test_examples.to(device)
        self.test_latent_reps = self.test_latent_reps.to(device)