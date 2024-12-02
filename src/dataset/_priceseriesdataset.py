import torch as th
from torch.utils.data import Dataset


class PriceSeriesDataset(Dataset):
    """
    A simple Dataset class for loading our using the torch APIs

    For simplicity, expects that the features are already segmented into windows
    of the desired sequence length. This could be adjusted in the future to allow
    the sequence length to be a parameter of the SequenceDataset class itself
    to abstract away the windowing process.
    """

    features: th.Tensor
    """The features of the dataset"""

    targets: th.Tensor
    """The targets of the dataset"""

    close_idx: int
    """The index of the close price column"""

    price_features: list[int]
    """The indices of the price features to normalize"""

    seq_len: int
    """The sequence length of the dataset windows"""

    feature_dim: int
    """The size of a single feature vector in the dataset"""

    t_0: float
    """The first time value in the dataset, a UNIX timestamp"""

    time_idx: th.Tensor
    """The time index of the dataset"""

    target_dist: th.Tensor
    """The distribution of the target labels"""

    target_counts: th.Tensor
    """The counts of each target label in the dataset"""


    def __init__(
            self,
            features: th.Tensor,
            targets: th.Tensor,
            price_features: list[int],
            close_idx: int = 5,
            seq_len: int = 10,
            t_0: float = None,
            name: str = "train",
            **kwargs
    ):
        """
        Initialize the dataset with the given features and targets.

        :param features: The time series features for the dataset.
        :param targets: The targets for the dataset at each time step.
        :param price_features: The indices of the price features to normalize.
        :param close_idx: The index of the close price column.
        :param seq_len: The sequence length of the dataset windows.
        :param t_0: The first time value in the dataset.
        :param name: The name of the dataset.
        """
        # Seq len can't be greater than the number of features
        assert seq_len <= features.shape[0], "Sequence length must be less than the number of features"
        assert seq_len > 0, "Sequence length must be greater than 0"
        assert price_features is not None, "Price features must be specified"

        self.close_idx = close_idx
        self.name = name

        # TODO save the actual prediction time index, not the first time index
        # TODO we want the time idx to represent the dates for which we're predicting
        # Don't use the last window as it will have no target
        self.t_0 = t_0 if t_0 is not None else features[0, 0].item()

        # We want a clean index that represents the dates for which we're predicting
        self.time_idx = features[seq_len-1:-1, 0].clone()

        # We subtract the first time value to get a relative time index starting at 0
        self.features = self.__create_seq_windows(features, seq_len, self.t_0)
        self.targets = targets[seq_len:]
        self.feature_dim = self.features.shape[-1]

        _, target_counts = th.unique(targets, return_counts=True)
        self.target_dist = target_counts / target_counts.sum()
        self.target_counts = target_counts

    @staticmethod
    def __create_seq_windows(
            features: th.Tensor,
            seq_len: int,
            time_offset: float
    ) -> th.Tensor:
        """
        Subtracts the first time value to get a relative time index.

        :param features: The features to create windows for.
        :param seq_len: The sequence length of the windows.
        :param time_offset: The first time value in the dataset.
        :return: A tensor of windows of the given sequence
        """
        features[:, 0] = (features[:, 0] - time_offset) / 86400  # seconds in a day
        return features.unfold(0, seq_len, 1)[:-1].transpose(1, 2)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
