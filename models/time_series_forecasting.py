import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesForecaster(nn.Module):
    def __init__(self, n_cont: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.lin = nn.Linear(128, 1)

    def forward(self, x):
        x = self.latent_representation(x)[0]
        x = self.lin(x)
        return x

    def latent_representation(self, x):
        x = self.lstm(x)
        return x

    def latent_to_output(self, x):
        x = self.lin(x)
        return x
