import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.base import BlackBox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MortalityGRU(BlackBox):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,  # dropout=drop_prob
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.latent_representation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = (
    #         weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
    #     )
    #     return hidden

    def latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        x, h = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x
