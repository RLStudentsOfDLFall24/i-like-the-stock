from collections import OrderedDict

import torch
from torch import nn

class TemporalAttentionLayer(nn.Module):
    """
    Implementation of a Temporal Attention Layer as first used by Fama & French
    We use the TAL to compress hidden states from  0 to t-1 into a single aggregate
    state.
    """

    hidden_dim: int
    """The dimension of the hidden states."""

    linear1: nn.Module
    """The first linear layer for the attention layer."""

    # bias_agg: nn.Parameter
    # """The bias for the aggregate state."""

    u_agg: nn.Parameter
    """The outer weight for the aggregate state."""

    def __init__(self, hidden_dim: int, agg_dim: int):
        super(TemporalAttentionLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.agg_dim = agg_dim

        self.weight_agg = nn.Linear(hidden_dim, agg_dim)
        self.layer_norm = nn.LayerNorm(agg_dim)
        self.u_agg = nn.Parameter(torch.zeros(agg_dim))

        nn.init.uniform_(self.u_agg)


    def forward(self, hidden_states):
        """
        Compute the aggregate state context vector from the hidden states.
        :param hidden_states: A batch of N x T x D hidden states.
        :return: A batch of N x (D + agg_dim) context vectors.
        """

        outputs = self.weight_agg(hidden_states)
        outputs = self.layer_norm(outputs)
        outputs = torch.tanh(outputs)
        outputs = torch.einsum("ntd,d->nt", outputs, self.u_agg)
        att_probs = torch.softmax(outputs, dim=1)
        agg_state = torch.einsum("nt,ntd->nd", att_probs, hidden_states)

        return torch.cat([agg_state, hidden_states[:, -1]], dim=1)