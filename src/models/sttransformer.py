import torch
from torch import nn, functional
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

    # periodic_activation: nn.Module = torch.sin
    # """The activation function for the periodic encoding, defaults to sin."""

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
        if self.input_dim > 1:
            nn.init.uniform_(self.frequencies, 0, 1)
            nn.init.uniform_(self.phase_shifts, 0, 1)

    def forward(self, inputs):
        """
        Create the time2vec encoding for the input data.

        :param inputs: A batch of N x T x F_t, sequences of time based features.
        :return: A batch of N x T x 2D encoding of the time feature data.
        """

        # Batch multiply each time feature with the frequencies
        outputs = torch.einsum("ntf,fd->ntd", inputs, self.frequencies)
        outputs += self.phase_shifts

        # We only apply sin to the periodic components, i.e. the last k-1 components
        outputs = torch.cat([outputs[:, :1], torch.sin(outputs[:, 1:])], dim=1)
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

        self.t2v = T2V(input_dim, n_frequencies)
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
        # new_dim = 1 + self.t2v.output_dim
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

    ctx_window: int
    """The length of the input sequences prior to encoding."""

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

    # TODO still not sure if we'll use LSTM module on top, but adding params for now
    num_lstm_layers: int
    """The number of LSTM layers."""

    lstm_dim: int
    """The dimension of the LSTM hidden state."""

    n_frequencies: int
    """The number of frequencies for the time2vec layer."""

    time_idx: list[int]
    """The index of the time feature in the input sequence, defaults to 0."""

    embedding: STEmbedding
    """The spatiotemporal embedding layer."""

    encoder_stack: nn.TransformerEncoder
    """The transformer encoder stack."""

    def __init__(
            self,
            d_features: int,
            device: torch.device,
            time_idx: list[int] = None,
            num_outputs: int = 3,
            num_encoders: int = 2,
            num_lstm_layers: int = 1,
            lstm_dim: int = 32,
            model_dim: int = 64,
            num_heads: int = 2,
            ctx_window: int = 10,
            n_frequencies: int = 32,
            fc_dim: int = 2048,
            fc_dropout: float = 0.1,
            **kwargs
    ):
        # Standard nn.Module initialization
        super(STTransformer, self).__init__()

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
        self.embedding = STEmbedding(d_features, model_dim, n_frequencies, time_idx)

        # 3. Setup the encoder stack with TransformerEncoderLayer
        self.encoder_stack = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=fc_dim,
                dropout=fc_dropout,
                batch_first=True
            ),
            num_layers=num_encoders
        )

        # TODO 4. Setup the LSTM layer

        # 5. Setup the output layer
        self.to(device)

    def forward(self, data):
        inputs = data.to(self.device)
        embedded = self.embedding(inputs)

        outputs = self.encoder_stack(embedded)

        print("what do we have")

        # Return a dummy tensor with the output shapes
        return torch.zeros(data.shape[0], self.num_outputs)


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed the random number generator
    torch.manual_seed(1984)
    # model = STTransformer(10)
    # Setup a simple sequence
    x_dim = 3
    seq_len = 10
    k = 4
    n = 5
    d_model = 64

    data = torch.randn(n, seq_len, x_dim)

    # embedding = STEmbedding(x_dim, d_model, k)
    #
    # out = embedding(data)
    # print(out)

    model = STTransformer(
        d_features=x_dim,
        device=device,
        num_encoders=2,
        num_outputs=1,
        num_lstm_layers=1,
        lstm_dim=32,
        time_idx=[0],
        model_dim=d_model,
        num_heads=2,
        ctx_window=10,
        n_frequencies=k,
        fc_dim=2048,
        fc_dropout=0.1
    )

    out = model(data)

    print('what')
    pass


if __name__ == '__main__':
    run()
