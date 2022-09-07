import torch


class Representer:
    def __init__(
        self,
        corpus_latent_reps: torch.Tensor,
        corpus_probas: torch.Tensor,
        corpus_true_classes: torch.Tensor,
        reg_factor: float,
    ) -> None:
        """
        Initialize a representer theorem explainer
        :param corpus_latent_reps: corpus latent representations
        :param corpus_probas: the probabilities predicted by the black-box for the corpus examples
        :param corpus_true_classes: the true classes associated to each corpus example
        :param reg_factor: the weight decay factor used in training the black-box model
        """
        self.corpus_latent_reps = corpus_latent_reps
        self.corpus_probas = corpus_probas
        self.corpus_true_classes = corpus_true_classes
        self.reg_factor = reg_factor
        self.corpus_size, self.dim_latent = corpus_latent_reps.shape
        self.num_classes = corpus_probas.shape[-1]
        self.test_latent_reps = None
        self.test_size = None
        self.weights = None

    def fit(self, test_latent_reps: torch.Tensor) -> None:
        """
        Fit the representer theorem explainer on test examples
        :param test_latent_reps: test example latent representations
        :return:
        """
        self.test_latent_reps = test_latent_reps
        self.test_size = test_latent_reps.shape[0]
        projections = torch.einsum(
            "ij,kj -> ik", test_latent_reps, self.corpus_latent_reps
        )
        projections = projections.view(self.test_size, self.corpus_size, 1)
        alpha = (self.corpus_true_classes - self.corpus_probas) / (
            2 * self.reg_factor * self.corpus_size
        )
        alpha = alpha.view(1, self.corpus_size, self.num_classes)
        self.weights = alpha * projections

    def output_approx(self) -> torch.Tensor:
        """
        Returns the representer theorem approximation of the test outputs
        :return:
        """
        output_approx = self.weights.sum(dim=1)
        return output_approx

    def to(self, device: torch.device) -> None:
        """
        Transfer the tensors to device
        :param device: the device where the tensors should be transferred
        :return:
        """
        self.corpus_latent_reps = self.corpus_latent_reps.to(device)
        self.corpus_probas = self.corpus_probas.to(device)
        self.corpus_true_classes = self.corpus_true_classes.to(device)
        self.test_latent_reps = self.test_latent_reps.to(device)
        self.weights = self.weights.to(device)
