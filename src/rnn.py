import torch
from torch import nn
from abstract_model import AbstractModel


class RNN(AbstractModel):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(10, 20)

    def forward(self, data):
        return self.rnn(data)

