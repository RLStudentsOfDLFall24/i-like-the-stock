import torch as th
from torch.utils.data import Dataset


class NextRowDataset(Dataset):
    """
    A basic dataset used for pretraining of the t2v module.

    The dataset is a single feature dataset where the prediction target for row
    k is row k+1.
"""

    features: th.Tensor
    """The features of the dataset"""

    targets: th.Tensor
    """The targets of the dataset"""

    def __init__(self, features: th.Tensor, targets: th.Tensor):
        self.features = features
        self.targets = targets

    def __len__(self):
        """
        Return the number of examples in this dataset.

        We have one less example than number of rows in the dataset as they are
        also the target values.
        """
        return self.features.shape[0]

    def __getitem__(self, idx) -> tuple[th.Tensor, th.Tensor]:
        """
        The target for row k is row k+1.
        :param idx: The index of the row to get.
        :return: The features for row k and row k+1.
        """
        return self.features[idx], self.targets[idx]

    @property
    def shape(self):
        return self.features.shape[0], self.features.shape[1]