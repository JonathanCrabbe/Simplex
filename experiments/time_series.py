from statsmodels.tsa.arima_process import arma_generate_sample
from visualization.time_series import plot_time_series
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from models.time_series_forecasting import TimeSeriesForecaster
import numpy as np
import torch
import matplotlib.pyplot as plt


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(TimeSeriesDataset).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_arma_old(random_seed: int = 42):
    ar_coefs = [1, .99]
    ma_coefs = [.0001, 0]
    np.random.seed(random_seed)
    X = arma_generate_sample(ar_coefs, ma_coefs, nsample=(10000, 31))
    X, Y = X[:, 10:-1], X[:, 11:]
    return X, Y


def generate_ar(ar_coefs: np.ndarray, random_seed: int = 42, length: int = 50, n_samples: int = 10000,
                variance: float = .1):
    np.random.seed(random_seed)
    p = len(ar_coefs)
    X = np.zeros((n_samples, length+1))
    X[:, :p] = variance*np.random.randn(n_samples, p)
    Noise = variance*np.random.randn(n_samples, length+1)

    # Better initialization
    for t in range(3*p):
        X_p = X[:, :p] @ ar_coefs[::-1] + variance*np.random.randn(n_samples)
        X[:, :p-1] = X[:, 1:p]
        X[:, p-1] = X_p

    for t in range(p, length+1):
        X[:, t] = X[:, t-p:t] @ ar_coefs[::-1] + Noise[:, t]

    return X[:, :-1], X[:, 1:]


def arma_precision(random_seed: int = 42, cv: int = 0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(random_seed)
    ar_coefs = np.array([.7, .25])
    length = 50
    n_samples = 10000

    X, Y = generate_ar(ar_coefs, random_seed + cv, length, n_samples)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed + cv)
    X_train = X_train.reshape(len(X_train), -1, 1)
    X_train = torch.from_numpy(X_train).float()
    X_test = X_test.reshape(len(X_test), -1, 1)
    X_test = torch.from_numpy(X_test).float().to(device)
    Y_train = Y_train.reshape(len(Y_train), -1, 1)
    Y_train = torch.from_numpy(Y_train).float()
    Y_test = Y_test.reshape(len(Y_test), -1, 1)
    Y_test = torch.from_numpy(Y_test).float().to(device)
    training_set = TimeSeriesDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=20, shuffle=True)

    model = TimeSeriesForecaster().to(device)
    opt = torch.optim.Adam(params=model.parameters())

    model.hidden = model.init_hidden(batch_size=len(X_test))
    print(f'Initial Test MSE: {torch.mean((Y_test - model(X_test))**2):.3g}')
    model.hidden = model.init_hidden(batch_size=1)
    plt.plot(np.arange(length), Y_test[2].detach().cpu().numpy())
    plt.plot(np.arange(length), model(X_test[2:3])[0].detach().cpu().numpy())
    plt.show()

    model.train()
    for epoch in range(30):
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            model.hidden = model.init_hidden(batch_size=len(X))
            opt.zero_grad()
            Y_pred = model(X)
            error = torch.sum((Y - Y_pred) ** 2)
            error.backward()
            opt.step()
        if (epoch+1) % 5 == 0:
            model.hidden = model.init_hidden(batch_size=len(X_test))
            print(f'Epoch {epoch + 1}: Test MSE = {torch.mean((Y_test - model(X_test)) ** 2):.3g}.')

    model.eval()
    model.hidden = model.init_hidden(batch_size=1)
    plt.plot(np.arange(length), Y_test[2].detach().cpu().numpy())
    plt.plot(np.arange(length), model(X_test[2:3])[0].detach().cpu().numpy())
    plt.show()


arma_precision()
