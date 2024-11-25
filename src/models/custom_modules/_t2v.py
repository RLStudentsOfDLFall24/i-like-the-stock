from collections import OrderedDict

import torch
from torch import nn


class T2V(nn.Module):
    """An implementation of time2Vec encoding for time series data."""

    input_dim: int
    """The number of features at each time step in a sequence."""

    n_frequencies: int
    """The number of frequencies to use in the output encoding."""

    linear_block: nn.Module
    """The linear layer for initial projection"""

    def __init__(
            self,
            input_dim: int,
            n_frequencies: int
    ):
        super(T2V, self).__init__()

        self.input_dim = input_dim
        self.output_dim = n_frequencies

        self.linear_block = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(input_dim, n_frequencies)),
            ("norm1", nn.LayerNorm(n_frequencies)),
            ("relu1", nn.GELU())
        ]))

    def forward(self, inputs, scaler: float = 1e8, log_t2v: bool = False):
        """
        Create the time2vec encoding for the input data.

        :param inputs: A batch of N x T x F_t, sequences of time based features.
        :param scaler: A scaling factor for the input data.
        :param log_t2v: Whether to log the time2vec encoding.
        :return: A batch of N x T x 2D encoding of the time feature data.
        """
        # Batch multiply each time feature with the frequencies
        outputs = self.linear_block(inputs)
        periodic_out = torch.sin(outputs[:, :, 1:])

        # We only apply sin to the periodic components, i.e. the last k-1 components
        outputs = torch.cat([outputs[:, :, :1], periodic_out], dim=-1)
        return outputs
