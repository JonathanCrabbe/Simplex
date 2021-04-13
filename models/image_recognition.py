import torch.nn as nn
import torch.nn.functional as F



class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def latent_representation(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x

    def forward(self, x):
        x = self.latent_representation(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

    def probabilities(self, x):
        x = self.latent_representation(x)
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

    def presoftmax(self, x):
        x = self.latent_representation(x)
        return self.fc2(x)

    def latent_to_presoftmax(self, h):
        return self.fc2(h)

