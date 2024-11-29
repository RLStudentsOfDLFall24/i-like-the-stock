from collections import OrderedDict

import torch
from torch import nn

from src.models.abstract_model import AbstractModel
from src.models.custom_modules import STEmbedding


class STTransformer(AbstractModel):
    """
    An implementation of the SpatioTemporal Transformer model for time series.
    """

    d_features: int
    """The number of features at each time step in a sequence."""

    device: torch.device
    """The device to run the model on."""

    model_dim: int
    """The dimension of the encoder stack."""

    num_encoders: int
    """The number of encoders in the stack."""

    num_heads: int
    """The number of heads in the multi-head attention layer."""

    fc_dim: int
    """The size of the fully connected layer."""

    fc_dropout: float
    """The dropout rate for the fully connected layer."""

    mlp_dim: int
    """The size of the MLP layer."""

    mlp_dropout: float
    """The dropout rate for the MLP layer."""

    num_outputs: int
    """The number of outputs in the output layer."""

    num_lstm_layers: int
    """The number of LSTM layers."""

    lstm_dim: int
    """The dimension of the LSTM hidden state."""

    n_frequencies: int
    """The number of frequencies for the time2vec layer."""

    seq_len: int
    """The context window for the transformer encoder."""

    time_idx: list[int]
    """The index of the time feature in the input sequence, defaults to 0."""

    embedding: STEmbedding
    """The spatiotemporal embedding layer."""

    encoder_stack: nn.TransformerEncoder | nn.Transformer
    """The transformer encoder stack."""

    def __init__(
            self,
            d_features: int,
            device: torch.device,
            time_idx: list[int] = None,
            num_outputs: int = 3,
            num_encoders: int = 2,
            num_lstm_layers: int = 1,
            lstm_dim: int = 64,
            model_dim: int = 64,
            num_heads: int = 2,
            n_frequencies: int = 32,
            fc_dim: int = 2048,
            fc_dropout: float = 0.1,
            mlp_dim: int = 2048,
            mlp_dropout: float = 0.4,
            seq_len: int = 32,
            batch_size: int = 32,
    ):
        # Standard nn.Module initialization
        super(STTransformer, self).__init__(batch_size=batch_size)

        # Assign the parameters to the class
        self.d_features = d_features
        self.device = device
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.num_encoders = num_encoders
        self.num_heads = num_heads
        self.fc_dim = fc_dim
        self.fc_dropout = fc_dropout
        self.num_outputs = num_outputs
        self.num_lstm_layers = num_lstm_layers
        self.lstm_dim = lstm_dim
        self.n_frequencies = n_frequencies
        self.time_idx = time_idx if time_idx is not None else [0]

        # Embedding layer -> outputs N x (seq_len * feature_dim) x D
        self.embedding = STEmbedding(
            d_features,
            model_dim,
            n_frequencies,
            time_idx
        )

        self.layer_norm = nn.LayerNorm(model_dim)

        # Use the transformer encoder stack to process the embedded data
        self.encoder_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=fc_dim,
                dropout=fc_dropout,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=num_encoders
        )
        # Add a batch norm layer to the LSTM output before the linear layer
        self.pre_lstm_layer_norm = nn.LayerNorm(model_dim)

        # We leverage an LSTM here to reduce to a single hidden state
        self.lstm = nn.LSTM(
            input_size=model_dim,
            hidden_size=lstm_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )


        # Linear output layers to classify the data
        self.linear = nn.Sequential(OrderedDict([
            ("fc_1", nn.Linear(in_features=lstm_dim, out_features=mlp_dim)),  # This layer the grad drops to 0.2-0.25
            ("fc_bn", nn.BatchNorm1d(mlp_dim)),
            ("fc_gelu", nn.GELU()),
            ("fc_drop", nn.Dropout(mlp_dropout)),
            ("fc_out", nn.Linear(in_features=mlp_dim, out_features=num_outputs))  # This layer has grad norms 0.3-0.4
        ]))

        self.to(device)

    def forward(self, data):
        inputs = data.to(self.device)
        embedded = self.embedding(inputs)
        # embedded = self.layer_norm(embedded)

        outputs = self.encoder_stack(embedded)
        # TODO: Investigate skip connection?
        # Can we norm here before the LSTM?
        outputs = self.pre_lstm_layer_norm(outputs)

        # outputs = outputs.reshape(inputs.shape[0], self.seq_len, -1)
        # Return a dummy tensor with the output shapes
        _, (h_t, _) = self.lstm(outputs)
        # TODO - add temporal attention layer

        # Apply a batch norm to the LSTM output
        # mlp_in = self.lstm_batch_norm(h_t[-1].squeeze(0))
        # Remember we have weird dims on h_t, so we can squeeze
        outputs = self.linear(h_t[-1].squeeze(0))
        # outputs = self.linear(mlp_in)
        return outputs
