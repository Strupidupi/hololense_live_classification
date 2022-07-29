import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, learning_rate,
     activation_function, l2_regularization_flag, l2_regularization_value):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.l2_regularization_flag = l2_regularization_flag
        self.l2_regularization_value = l2_regularization_value

        self.criterion = nn.CrossEntropyLoss()
        if l2_regularization_flag:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=l2_regularization_value)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        out = F.log_softmax(out, dim=1)
        out = torch.exp(out)
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        if torch.cuda.is_available():
            return [t.cuda() for t in (h0, c0)]
        else:
            return [t for t in (h0, c0)]
            

    def save(self, path):
        torch.save(self.state_dict(), path+'/model.pth')

    def get_optimizer(self):
        return self.optimizer