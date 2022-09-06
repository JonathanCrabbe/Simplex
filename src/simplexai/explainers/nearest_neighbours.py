import torch
from sklearn.neighbors import KNeighborsRegressor


class NearNeighLatent:
    def __init__(
        self,
        corpus_examples: torch.Tensor,
        corpus_latent_reps: torch.Tensor,
        weights_type: str = "uniform",
    ) -> None:
        """
        Initialize a latent nearest neighbours explainer
        :param corpus_examples: corpus input features
        :param corpus_latent_reps: corpus latent representations
        :param weights_type: type of KNN weighting (uniform or distance)
        """
        self.corpus_examples = corpus_examples
        self.corpus_latent_reps = corpus_latent_reps
        self.corpus_size = corpus_examples.shape[0]
        self.dim_latent = corpus_latent_reps.shape[-1]
        self.weights_type = weights_type
        self.n_test = None
        self.test_examples = None
        self.test_latent_reps = None
        self.regressor = None

    def fit(
        self,
        test_examples: torch.Tensor,
        test_latent_reps: torch.Tensor,
        n_keep: int = 5,
    ) -> None:
        """
        Fit the nearest neighbour explainer on test examples
        :param test_examples: test example input features
        :param test_latent_reps: test example latent representations
        :param n_keep: number of neighbours used to build a latent decomposition
        :return:
        """
        regressor = KNeighborsRegressor(n_neighbors=n_keep, weights=self.weights_type)
        regressor.fit(
            self.corpus_latent_reps.clone().detach().cpu().numpy(),
            self.corpus_latent_reps.clone().detach().cpu().numpy(),
        )
        self.regressor = regressor
        self.test_examples = test_examples
        self.n_test = test_examples.shape[0]
        self.test_latent_reps = test_latent_reps

    def latent_approx(self) -> torch.Tensor:
        """
        Returns the latent approximation of test_latent_reps with the nearest corpus neighbours
        :return: approximate latent representations as a tensor
        """
        approx_reps = self.regressor.predict(
            self.test_latent_reps.clone().detach().cpu().numpy()
        )
        return (
            torch.from_numpy(approx_reps)
            .type(torch.float32)
            .to(self.test_latent_reps.device)
        )

    def to(self, device: torch.device) -> None:
        """
        Transfer the tensors to device
        :param device: the device where the tensors should be transferred
        :return:
        """
        self.corpus_examples = self.corpus_examples.to(device)
        self.corpus_latent_reps = self.corpus_latent_reps.to(device)
        self.test_examples = self.test_examples.to(device)
        self.test_latent_reps = self.test_latent_reps.to(device)
