import torch
from torch import nn
from .abstract_model import AbstractModel


class Transformer(AbstractModel):
    def __init__(self, batch_size):
        super(Transformer, self).__init__(batch_size=batch_size)
        self.model = nn.RNN(batch_size, 20)

    def forward(self, data):
        return super().forward(data)

