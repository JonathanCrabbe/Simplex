import pandas as pd
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class ProstateCancerDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = torch.tensor(self.X.iloc[i, :])
        if self.y is not None:
            return data, torch.tensor(self.y.iloc[i])
        else:
            return data


def load_seer():
    features = ['age', 'psa', 'comorbidities', 'treatment_CM', 'treatment_Primary hormone therapy',
                'treatment_Radical Therapy-RDx', 'treatment_Radical therapy-Sx', 'grade_1.0', 'grade_2.0', 'grade_3.0',
                'grade_4.0', 'grade_5.0', 'stage_1', 'stage_2', 'stage_3', 'stage_4', 'gleason1_1', 'gleason1_2',
                'gleason1_3', 'gleason1_4', 'gleason1_5', 'gleason2_1', 'gleason2_2', 'gleason2_3', 'gleason2_4',
                'gleason2_5']
    label = ['mortCancer']
    df = pd.read_csv('./data/Prostate Cancer/seer_external_imputed_new.csv')
    return df[features], df[label]


X, y = load_seer()
cont_dims = [0, 1, 2]  # Only the first 3 features are continuous
data = ProstateCancerDataset(X, y)
data_loader = DataLoader(data, batch_size=128, shuffle=True)
for x, y in data_loader:
    print(x[0])

