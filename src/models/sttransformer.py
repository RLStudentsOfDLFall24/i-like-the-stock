import torch
from torch import nn

from src.models.abstract_model import AbstractModel


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
        self.frequencies = nn.Parameter(torch.zeros((input_dim, n_frequencies)))
        self.phase_shifts = nn.Parameter(torch.zeros(n_frequencies))
        # self.norm_layer = nn.BatchNorm1d(n_frequencies)
        self.norm_layer = nn.LayerNorm(n_frequencies)

        nn.init.uniform_(self.frequencies, 0, 2 * torch.pi)
        nn.init.uniform_(self.phase_shifts, 0, 2 * torch.pi)

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
        outputs = inputs.view(n, t, f_t, 1) * self.frequencies.view(1, 1, f_t, -1)
        # outputs = torch.einsum("ntf,fd->ntd", scaled_inputs, self.frequencies)

        # Todo - should this come after the sin?
        # outputs = self.norm_layer(outputs)

        # We can batch norm here I think - or Layer norm w/e makes the most sense
        # The values are way too large here
        outputs += self.phase_shifts

        # We only apply sin to the periodic components, i.e. the last k-1 components
        outputs = torch.cat([outputs[:, :1], torch.sin(outputs[:, 1:])], dim=1)

        # Here we want to sum the signals across times so we can return N x T x 2D
        outputs = outputs.sum(dim=2)
        return outputs


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

        # 2. We expand the encoded sequence and the data
        n, t, d = inputs.shape
        data_expanded = inputs.view(n, t, d, 1)
        encoded_expand = encoded.unsqueeze(2).expand(-1, -1, d, -1)

        # Concat the expanded data and encoded sequence on last dim
        concatenated = torch.cat([data_expanded, encoded_expand], dim=-1).view(n, -1, self.dense_input_size)

        # Pass through the dense
        return self.dense(concatenated)



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

    num_outputs: int
    """The number of outputs in the output layer."""

    num_lstm_layers: int
    """The number of LSTM layers."""

    lstm_dim: int
    """The dimension of the LSTM hidden state."""

    n_frequencies: int
    """The number of frequencies for the time2vec layer."""

    ctx_window: int
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
            ctx_window: int = 32,
            batch_size: int = 32,
            **kwargs
    ):
        # Standard nn.Module initialization
        super(STTransformer, self).__init__(batch_size=batch_size)

        # Assign the parameters to the class
        self.d_features = d_features
        self.device = device
        self.ctx_window = ctx_window
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

        # We leverage an LSTM here to reduce to a single hidden state
        self.lstm = nn.LSTM(
            input_size=d_features * model_dim,
            hidden_size=lstm_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Add a batch norm layer to the LSTM output before the linear layer
        self.lstm_batch_norm = nn.BatchNorm1d(lstm_dim)

        # Linear output layers to classify the data
        self.linear = nn.Sequential(
            nn.Linear(in_features=lstm_dim, out_features=fc_dim),
            nn.GELU(),
            nn.Linear(in_features=fc_dim, out_features=num_outputs)
        )

        self.to(device)

    def forward(self, data):
        inputs = data.to(self.device)
        embedded = self.embedding(inputs)
        embedded = self.layer_norm(embedded)

        outputs = self.encoder_stack(embedded)
        outputs = outputs.view(inputs.shape[0], self.ctx_window, -1)

        # Return a dummy tensor with the output shapes
        _, (h_t, _) = self.lstm(outputs)

        # Apply a batch norm to the LSTM output
        mlp_in = self.lstm_batch_norm(h_t[-1].squeeze(0))
        # Remember we have weird dims on h_t, so we can squeeze
        outputs = self.linear(mlp_in)
        return outputs
