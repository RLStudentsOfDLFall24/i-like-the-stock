from collections import OrderedDict

import torch
from torch import nn


class T2V(nn.Module):
    """An implementation of time2Vec encoding for time series data."""

    input_dim: int
    """The number of features at each time step in a sequence."""

    n_frequencies: int
    """The number of frequencies to use in the output encoding."""

    periodic_block: nn.Module
    """The linear layer for initial projection"""

    def __init__(
            self,
            input_dim: int = 3,
            n_frequencies: int = 64,
    ):
        super(T2V, self).__init__()

        self.input_dim = input_dim
        self.output_dim = n_frequencies

        # We have one non-periodic component and k-1 periodic components
        self.non_periodic_block = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(input_dim, 1)),
        ]))
        self.periodic_block = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(input_dim, n_frequencies - 1)),
        ]))

    def forward(self, inputs):
        """
        Encode the input data using the t2v learned embedding.

        :param inputs: A batch of N x T x F_t, sequences of time based features.
        :return: A batch of N x T x 2D encoding of the time feature data.
        """
        # We're only using pretrained weights for year / month / day
        assert inputs.shape[-1] == 3, "Only year / month / day are supported"

        # Batch multiply each time feature with the frequencies
        non_periodic = self.non_periodic_block(inputs)
        periodics = torch.sin(self.periodic_block(inputs))

        # We only apply sin to the periodic components, i.e. the last k-1 components
        outputs = torch.cat([non_periodic, periodics], dim=-1)
        return outputs
