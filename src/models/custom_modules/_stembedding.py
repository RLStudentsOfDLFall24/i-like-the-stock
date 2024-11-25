import torch
from torch import nn

from ._t2v import T2V

class STEmbedding(nn.Module):
    """
    Implementation of a spatiotemporal feature embedding layer.

    Uses a time2vec encoding for the time feature before passing through a dense
    layer to produce the input features.
    """

    input_dim: int
    """The number of features at each time step in a sequence."""

    output_dim: int
    """The dimension of the output embeddings."""

    n_frequencies: int
    """The number of frequencies for the time2vec layer."""

    time_idx: list[int]
    """The index of the time feature(s) in the input sequence."""

    t2v: T2V
    """The time2vec encoding layer."""

    dense_input_size: int
    """The size of the input to the dense layer."""

    dense: nn.Linear
    """The dense layer for producing embeddings."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_frequencies: int,
            time_idx: list[int] = None
    ):
        super(STEmbedding, self).__init__()

        self.t2v = T2V(len(time_idx), n_frequencies)
        self.dense_input_size = 1 + n_frequencies
        self.dense = nn.Linear(self.dense_input_size, output_dim)
        self.time_idx = time_idx if time_idx is not None else [0]

    def forward(self, inputs):
        """
        Forward pass for the spatiotemporal embedding layer.

        :param inputs: A batch of N x T x F_t, sequences of time based features.
        :return: A batch of N x T x D embeddings of the input data.
        """
        # 1. Encode the time features using time2vec
        encoded = self.t2v(inputs[:, :, self.time_idx])

        # TODO 1b. Project numeric features for time to the same dimension as the encoded features

        # 2. We expand the encoded sequence and the data
        n, t, d = inputs.shape
        data_expanded = inputs.reshape(n, t, d, 1)
        encoded_expand = encoded.unsqueeze(2).expand(-1, -1, d, -1)

        # Concat the expanded data and encoded sequence on last dim
        concatenated = torch.cat([data_expanded, encoded_expand], dim=-1).reshape(n, -1, self.dense_input_size)

        # Pass through the dense
        return self.dense(concatenated)
