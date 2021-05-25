import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesForecaster(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 100, output_dim: int = 1, num_layers: int = 2,
                 batch_size: int = 20):
        super(TimeSeriesForecaster, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                            batch_first=True, num_layers=self.num_layers)
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden(batch_size)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        '''
         self.rnn = nn.RNN(input_size=self.input_dim, hidden_size=self.hidden_dim,
                          batch_first=True, num_layers=self.num_layers)
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                            batch_first=True, num_layers=self.num_layers)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
                
                        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        '''

    def init_hidden(self, batch_size):
        h0, c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim),\
                 torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return h0, c0

    def forward(self, x):
        x, self.hidden = self.lstm(x)
        x = self.lin(x)
        return x

    def latent_representation(self, x):
        x, self.hidden = self.lstm(x)
        return x[:, -1, :]

    def latent_to_output(self, x):
        x = self.lin(x)
        return x
