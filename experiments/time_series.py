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


def generate_arma(random_seed: int = 42):
    ar_coefs = [1, -.9]
    ma_coefs = [1, .25, .15]
    np.random.seed(random_seed)
    X = arma_generate_sample(ar_coefs, ma_coefs, nsample=(10000, 31))
    X, Y = X[:, 10:-1], X[:, 11:]
    return X, Y


def arma_precision(random_seed: int = 42, cv: int = 0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(random_seed)
    X, Y = generate_arma(random_seed + cv)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed + cv)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    Y_train = scaler.transform(Y_train)
    X_train = X_train.reshape(len(X_train), -1, 1)
    X_train = torch.from_numpy(X_train).float()
    X_test = scaler.transform(X_test)
    Y_test = scaler.transform(Y_test)
    X_test = X_test.reshape(len(X_test), -1, 1)
    X_test = torch.from_numpy(X_test).float().to(device)
    Y_train = Y_train.reshape(len(Y_train), -1, 1)
    Y_train = torch.from_numpy(Y_train).float()
    Y_test = Y_test.reshape(len(Y_test), -1, 1)
    Y_test = torch.from_numpy(Y_test).float().to(device)
    training_set = TimeSeriesDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=20, shuffle=True)

    #plot_time_series(X_train[0])
    #plot_time_series(X_test[0])

    model = TimeSeriesForecaster().to(device)
    opt = torch.optim.Adam(params=model.parameters(), lr=1.0e-3)

    print(torch.mean(torch.abs(Y_test - model(X_test))))
    model.hidden = model.init_hidden(batch_size=1)
    plt.plot(np.arange(0, X_test.shape[1]), X_test[0].detach().cpu().numpy())
    plt.plot(np.arange(1, X_test.shape[1]+1), model(X_test[0:1])[0].detach().cpu().numpy())
    plt.show()

    model.train()
    for epoch in range(100):
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            model.hidden = model.init_hidden(batch_size=len(X))
            opt.zero_grad()
            Y_pred = model(X)
            error = torch.sum((Y - Y_pred) ** 2)
            error.backward()
            opt.step()
        if epoch % 2 == 0:
            model.hidden = model.init_hidden(batch_size=len(X_test))
            print(f'Epoch {epoch+1}: Test MSE = {torch.mean((Y_test - model(X_test))**2):.3g}.')

    model.hidden = model.init_hidden(batch_size=1)
    plt.plot(np.arange(0, X_test.shape[1]), X_test[0].detach().cpu().numpy())
    plt.plot(np.arange(1, X_test.shape[1] + 1), model(X_test[0:1])[0].detach().cpu().numpy())
    plt.show()


arma_precision()
