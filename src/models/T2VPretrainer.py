import torch
from torch import nn

from .custom_modules import T2V

class T2VPretrainer(nn.Module):

    t2v: T2V
    """The time2vec encoding layer."""

    estimator: nn.Module
    """Output network used for the pretraining task."""

    device: torch.device
    """The device to run the model on."""

    def __init__(
            self,
            input_dim: int,
            n_frequencies: int,
            mlp_dim: int = 256,
            device: torch.device = torch.device("cpu")
    ):
        super(T2VPretrainer, self).__init__()
        self.device = device
        self.t2v = T2V(input_dim, n_frequencies)

        # We match the output dim to the input dimensions
        self.mlp_estimator = nn.Sequential(
            nn.Linear(n_frequencies, mlp_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(mlp_dim, input_dim)
        )

        self.to(self.device)

    def forward(self, inputs):
        """
        Encode the input data using the t2v learned embedding.

        :param inputs: A batch of N x T x input_dim, sequences of time based features.
        :return: A batch of N x T x input_dim predictions for the target data.
        """
        t2v_embeddings = self.t2v(inputs.to(self.device))
        return self.mlp_estimator(t2v_embeddings)
