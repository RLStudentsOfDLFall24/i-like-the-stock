import torch
from torch import nn
from .abstract_model import AbstractModel


class RNN(AbstractModel):
    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        super(RNN, self).__init__(batch_size=batch_size)
        self.model = nn.RNN(batch_size, 20)

    def forward(self, data):
        return super().forward(data)

