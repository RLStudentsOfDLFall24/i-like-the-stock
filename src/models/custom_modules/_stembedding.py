import torch
from torch import nn

from ._naivet2v import NaiveT2V
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

    t2v: nn.Module
    """The time2vec encoding layer."""

    numeric_projection: nn.Module
    """The projection layer for numeric features."""

    numeric_idx: list[int]
    """The index of the numeric features in the input sequence."""

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

        n_time_features = len(time_idx)

        # Encode the time, project the numeric features to the same dimension
        # TODO make time index mandatory parameter
        self.n_frequencies = n_frequencies
        # self.t2v = T2V(n_time_features - 1, n_frequencies)
        self.t2v = NaiveT2V(n_frequencies)

        # # Try projecting the numeric features to the same dimension as the encoded time features
        # self.numeric_projection = nn.Sequential(
        #     nn.Linear(input_dim - n_time_features, n_frequencies),
        #     nn.LayerNorm(n_frequencies),
        #     nn.GELU()
        # )

        # self.dense_input_size = 2 * n_frequencies # Concat dims
        self.dense_input_size = 1 + n_frequencies

        self.dense = nn.Linear(self.dense_input_size, output_dim) # concat dims
        self.time_idx = time_idx if time_idx is not None else [0]
        self.numeric_idx = [i for i in range(input_dim) if i not in self.time_idx]

    def forward(self, inputs):
        """
        Forward pass for the spatiotemporal embedding layer.

        :param inputs: A batch of N x T x F_t, sequences of time based features.
        :return: A batch of N x T x D embeddings of the input data.
        """
        # 1. Encode the time features using time2vec
        # Ignore the 0 index in time # Testing
        actual_time_idx = [i for i in self.time_idx if i != 0]
        encoded = self.t2v(inputs[:, :, actual_time_idx])

        # TODO - try using a pretrained t2v model

        # TODO 1b. Project numeric features for time to the same dimension as the encoded features
        # num_projected = self.numeric_projection(inputs[:, :, self.numeric_idx])

        # TODO Fix the ST Transformer to expect the new shape
        # # 2. We expand the encoded sequence and the data # Might revisit
        n, t, d = inputs.shape
        data_expanded = inputs.reshape(n, t, d, 1)
        encoded_expand = encoded.unsqueeze(2).expand(-1, -1, d, -1)

        # Concat the expanded data and encoded sequence on last dim
        concatenated = torch.cat([data_expanded, encoded_expand], dim=-1).reshape(n, -1, self.dense_input_size)
        # concatenated = torch.cat([encoded, num_projected], dim=-1)

        # Pass through the dense
        return self.dense(concatenated) # If using concatenated TODO probably use this
