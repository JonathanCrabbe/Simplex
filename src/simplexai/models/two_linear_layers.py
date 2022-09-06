import torch
import torch.nn as nn
import torch.nn.functional as F

from simplexai.models.base import BlackBox


class TwoLayerMortalityPredictor(BlackBox):
    def __init__(self, n_cont: int = 3, input_feature_num=26) -> None:
        """
        Mortality predictor MLP
        :param n_cont: number of continuous features among the output features
        """
        super().__init__()
        self.n_cont = n_cont
        self.lin1 = nn.Linear(input_feature_num, 50)
        self.lin2 = nn.Linear(50, 2)
        self.bn1 = nn.BatchNorm1d(self.n_cont)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.latent_representation(x)
        x = self.lin2(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x_cont, x_disc = x[:, : self.n_cont], x[:, self.n_cont :]
        x_cont = self.bn1(x_cont)
        x = torch.cat([x_cont, x_disc], 1)
        x = self.lin1(x)
        return x

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the class probabilities for the input x
        :param x: input features
        :return: probabilities
        """
        x = self.latent_representation(x)
        x = self.lin2(x)
        x = F.softmax(x, dim=-1)
        return x

    def latent_to_presoftmax(self, h: torch.Tensor) -> torch.Tensor:
        """
        Maps a latent representation to a preactivation output
        :param h: latent representations
        :return: presoftmax activations
        """
        h = self.lin2(h)
        return h
