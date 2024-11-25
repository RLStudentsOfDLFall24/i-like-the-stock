import torch
from torch import nn


class T2V(nn.Module):
    """An implementation of time2Vec encoding for time series data."""

    input_dim: int
    """The number of features at each time step in a sequence."""

    n_frequencies: int
    """The number of frequencies to use in the output encoding."""

    frequencies: nn.Parameter
    """The frequencies for the encoding."""

    phase_shifts: nn.Parameter
    """The phase shifts for the encoding."""

    norm_layer: nn.BatchNorm1d | nn.LayerNorm
    """Batch norm layer for the linear output of the frequencies."""

    def __init__(
            self,
            input_dim: int,
            n_frequencies: int
    ):
        super(T2V, self).__init__()

        self.input_dim = input_dim
        self.output_dim = n_frequencies

        # Create the frequencies and phase shifts for the encoding
        self.frequencies = nn.Parameter(torch.zeros((input_dim, n_frequencies)), requires_grad=True)
        self.phase_shifts = nn.Parameter(torch.zeros(n_frequencies), requires_grad=True)
        # self.norm_layer = nn.BatchNorm1d(n_frequencies)
        self.norm_layer = nn.LayerNorm(n_frequencies)

        nn.init.uniform_(self.frequencies)
        nn.init.uniform_(self.phase_shifts)

    def forward(self, inputs, scaler: float = 1e4, log_t2v: bool = False):
        """
        Create the time2vec encoding for the input data.

        :param inputs: A batch of N x T x F_t, sequences of time based features.
        :param scaler: A scaling factor for the input data.
        :param log_t2v: Whether to log the time2vec encoding.
        :return: A batch of N x T x 2D encoding of the time feature data.
        """
        # we need to scale the timestamp only for index 0
        n, t, f_t = inputs.shape
        scaled_inputs = inputs.clone()
        scaled_inputs[:, :, 0] = scaled_inputs[:, :, 0] / scaler

        # Batch multiply each time feature with the frequencies
        # TODO I think this is supposed to be a hadamard product
        # outputs = scaled_inputs.reshape(n, t, f_t, 1) * self.frequencies.reshape(1, 1, f_t, -1)
        outputs = torch.einsum("ntf,fd->ntd", scaled_inputs, self.frequencies)

        # Todo - should this come after the sin?
        # outputs = self.norm_layer(outputs)
        # We can batch norm here I think - or Layer norm w/e makes the most sense
        # The values are way too large here
        outputs += self.phase_shifts

        # We only apply sin to the periodic components, i.e. the last k-1 components
        outputs = torch.cat([outputs[:, :1], torch.sin(outputs[:, 1:])], dim=1)

        # # Here we want to sum the signals across times so we can return N x T x 2D
        # outputs = outputs.sum(dim=2)
        return outputs
