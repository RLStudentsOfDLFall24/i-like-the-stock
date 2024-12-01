from collections import OrderedDict

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

    ignore_cols: list[int]
    """The index of the columns to ignore in the input sequence."""

    t2v: nn.Module
    """The time2vec encoding layer."""

    feature_idx: list[int]
    """The index of the features in the input sequence."""

    dense_input_size: int
    """The size of the input to the dense layer."""

    dense: nn.Module
    """The dense layer for producing embeddings."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            time_idx: list[int] = None,
            ignore_cols: list[int] = None,
            n_frequencies: int = 64,
            pretrained_t2v: str = None,
    ):
        super(STEmbedding, self).__init__()

        n_time_features = len(time_idx)
        self.n_frequencies = n_frequencies

        self.t2v = T2V(n_time_features, n_frequencies)
        if pretrained_t2v is None:
            raise ValueError("Pretrained time2vec model must be specified")

        # Load and freeze T2V model weights
        self.t2v.load_state_dict(torch.load(pretrained_t2v, weights_only=True))
        for param in self.t2v.parameters():
            param.requires_grad = False

        # TODO allow for expanded or condensed output.
        # TODO if condensed - we pass the time features through a dense layer
        # TODO then we concatenate the time features with the projection
        # TODO the output from this representation will be 2 * n_frequencies
        # TODO this should only change the input size to the dense output layer
        self.dense_input_size = 1 + n_frequencies

        self.dense = nn.Sequential(
            OrderedDict([
                ("ste_fc1", nn.Linear(self.dense_input_size, output_dim)),
                ("ste_gelu1", nn.GELU()),
                ("ste_fc2", nn.Linear(output_dim, output_dim)),
            ]))

        # Ortho init for the dense layer
        nn.init.orthogonal_(self.dense[0].weight)
        nn.init.orthogonal_(self.dense[2].weight)

        # Housekeeping on indices
        self.time_idx = time_idx if time_idx is not None else [0]
        self.ignore_cols = ignore_cols if ignore_cols is not None else []
        self.feature_idx = [
            i for i in range(input_dim)
            if i not in self.time_idx and i not in self.ignore_cols
        ]

    def forward(self, inputs):
        """
        Forward pass for the spatiotemporal embedding layer.

        :param inputs: A batch of N x T x F_t, sequences of time based features.
        :param ignore_cols: A list of columns to ignore in the input data.
        :return: A batch of N x T x D embeddings of the input data.
        """
        time_encoded = self.t2v(inputs[:, :, self.time_idx])
        num_features = inputs[:, :, self.feature_idx]

        n, t, d = num_features.shape
        data_expanded = num_features.reshape(n, t, d, 1)
        encoded_expand = time_encoded.unsqueeze(2).expand(-1, -1, d, -1)

        # Concat the expanded data and encoded sequence on last dim
        concatenated = torch.cat([data_expanded, encoded_expand], dim=-1).reshape(n, -1, self.dense_input_size)
        return self.dense(concatenated)
