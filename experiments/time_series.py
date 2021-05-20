from statsmodels.tsa.arima_process import arma_generate_sample
from visualization.time_series import plot_time_series
from sklearn.model_selection import train_test_split
from models.time_series_forecasting import TimeSeriesForecaster
import numpy as np
import torch


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(TimeSeriesDataset).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_arma(random_seed: int = 42):
    ar_coefs = [1, .75, -.25]
    ma_coefs = [1, .65, .35]
    np.random.seed(random_seed)
    X = arma_generate_sample(ar_coefs, ma_coefs, nsample=(1000, 50))
    X, y = torch.from_numpy(X[:, :-1]), torch.from_numpy(X[:, -1])
    X = X.view(1000, 49, 1)
    return X.float(), y.float()


def arma_precision(random_seed: int = 42, cv: int = 0):
    torch.manual_seed(random_seed)
    X, y = generate_arma(random_seed+cv)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed+cv)
    training_set = TimeSeriesDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=20, shuffle=True)

    model = TimeSeriesForecaster()
    opt = torch.optim.Adam(params=model.parameters())

    model.train()
    for epoch in range(10):
        for X, y in train_loader:
            opt.zero_grad()
            y_pred = model(X)
            error = torch.mean((y-y_pred)**2)
            error.backward()
            opt.step()

    print(torch.mean((y_test - model(X_test))**2))



arma_precision()