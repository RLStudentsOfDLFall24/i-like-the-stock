import torch
from torch import nn
from .abstract_model import AbstractModel


class RNN(AbstractModel):
    def __init__(self,
                 d_features: int,
                 batch_size: int,
                 seq_len: int,
                 num_layers: int,
                 device: torch.device):
        super(RNN, self).__init__(batch_size=batch_size)
        self.model = nn.RNN(batch_size, 20)

    def forward(self, data):
        #import pdb; pdb.set_trace()
        return super().forward(data)

