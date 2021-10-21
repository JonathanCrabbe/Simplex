import abc
import torch


class BlackBox(torch.nn.Module):
    @abc.abstractmethod
    def latent_representation(self, x) -> torch.Tensor:
        return

    @abc.abstractmethod
    def forward(self, x) -> torch.Tensor:
        return
