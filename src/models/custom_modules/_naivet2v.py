from collections import OrderedDict

import torch
from torch import nn

class NaiveT2V(nn.Module):
    n_frequenices: int
    """The number of frequencies to use in the output encoding."""
    def __init__(self, n_frequencies: int):
        super(NaiveT2V, self).__init__()

        self.n_frequencies = n_frequencies
        # In the year 3000...
        self.year_embedding = nn.Embedding(3000, n_frequencies)
        self.month_embedding = nn.Embedding(13, n_frequencies)
        self.day_embedding = nn.Embedding(32, n_frequencies)

    def forward(self, inputs, scaler: float = 1e8, log_t2v: bool = False):
        """Encode year / month / day -> then sum them"""
        try:
            assert (inputs[:, :, 0] >= 0).all() and (inputs[:, :, 0] <= 3000).all(), "Year out of bounds"
            assert (inputs[:, :, 1] >= 0).all() and (inputs[:, :, 1] < 13).all(), "Month out of bounds"
            assert (inputs[:, :, 2] >= 0).all() and (inputs[:, :, 2] < 32).all(), "Day out of bounds"

            year = self.year_embedding(inputs[:, :, 0].long())
            month = self.month_embedding(inputs[:, :, 1].long())
            day = self.day_embedding(inputs[:, :, 2].long())

            return year + month + day
        except AssertionError as e:
            print(e)
            raise
