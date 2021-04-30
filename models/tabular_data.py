import torch
import torch.nn as nn
import torch.nn.functional as F


class MortalityPredictor(nn.Module):
    def __init__(self, n_cont: int = 3):
        super().__init__()
        self.n_cont = n_cont
        self.lin1 = nn.Linear(26, 200)
        self.lin2 = nn.Linear(200, 50)
        self.lin3 = nn.Linear(50, 2)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(50)
        self.drops = nn.Dropout(0.3)

    def forward(self, x):
        x = self.latent_representation(x)
        x = self.lin3(x)
        x = F.log_softmax(x, dim=-1)
        return x

    def latent_representation(self, x):
        x_cont, x_disc = x[:, :self.n_cont], x[:, self.n_cont:]
        x_cont = self.bn1(x_cont)
        x = torch.cat([x_cont, x_disc], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        # x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        # x = self.bn3(x)
        return x

    def probabilities(self, x):
        x = self.latent_representation(x)
        x = self.lin3(x)
        x = F.softmax(x, dim=-1)
        return x

    def latent_to_presoftmax(self, x):
        x = self.lin3(x)
        return x
