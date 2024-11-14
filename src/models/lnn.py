import torch
from torch import nn
from .abstract_model import AbstractModel


class LNN(AbstractModel):
    def __init__(self):
        super().__init__()
        self.model = nn.RNN(10, 20)

    def forward(self, data):
        return super().forward(data)

