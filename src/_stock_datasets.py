"""
@Author: Sakae Watanabe
@Date: 2024-11-14
@Description: Utilities for loading and processing of train/test data.


We export the following functions and classes via __init__.py:
    - PriceSeriesDataset
    - create_datasets

Example importing from top level:
    from src import create_datasets, PriceSeriesDataset
"""
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset


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

    seq_len: int
    """The sequence length of the dataset windows"""

    def __init__(self, features: th.Tensor, targets: th.Tensor, close_idx: int = 5, seq_len: int = 10):
        """
        Initialize the dataset with the given features and targets.

        :param features: The time series features for the dataset.
        :param targets: The targets for the dataset at each time step.
        :param close_idx: The index of the close price column.
        :param seq_len: The sequence length of the dataset windows.
        """
        # Seq len can't be greater than the number of features
        assert seq_len <= features.shape[0], "Sequence length must be less than the number of features"
        assert seq_len > 0, "Sequence length must be greater than 0"

        # We normalize price values by the first value of adj_close (col 5)
        initial_price = features[0, close_idx]
        features[:, 1:6] = features[:, 1:6] / initial_price

        # Don't use the last window as it will have no target
        self.features = features.unfold(0, seq_len, 1)[:-1].transpose(1, 2)
        self.targets = targets[seq_len:]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def load_symbol(
        symbol: str,
        root: str = "../data/clean",
        target_type: str = "basic"
) -> tuple[th.Tensor, th.Tensor]:
    """
    Load the data for the given symbol into tensors.

    :param symbol: The symbol to load the data for.
    :param root: The root directory to load the data from.
    :param target_type: The type of target to load, default "basic".
    :return: A tuple of torch.Tensors containing the features and targets
    """
    data = np.genfromtxt(
        f"{root}/{symbol}.csv",
        delimiter=",",
        skip_header=1,
    )

    targets = np.genfromtxt(
        f"{root}/{symbol}_target_{target_type}.csv",
        dtype=int,
        delimiter=","
    )
    # Split the features and targets
    features = th.tensor(data)
    targets = th.tensor(targets)

    return features, targets


def create_splits(
        features: th.Tensor,
        targets: th.Tensor,
        train_size: float = 0.75,
        val_size: float = 0.15,
        seq_len: int = 10
) -> tuple[PriceSeriesDataset, PriceSeriesDataset, PriceSeriesDataset]:
    """
    Split the data into train/valid/test sets and create DataLoader objects.
    """
    # Get counts for splits
    n_rows = features.shape[0]
    n_train, n_val = int(n_rows * train_size), int(n_rows * val_size)
    n_test = n_rows - n_train - n_val

    x_train, x_valid, x_test = th.split(features, [n_train, n_val, n_test])
    y_train, y_valid, y_test = th.split(targets, [n_train, n_val, n_test])

    # Create the datasets
    return (
        PriceSeriesDataset(x_train, y_train, seq_len=seq_len),
        PriceSeriesDataset(x_valid, y_valid, seq_len=seq_len),
        PriceSeriesDataset(x_test, y_test, seq_len=seq_len)
    )


def create_datasets(
        symbol: str,
        root: str = "../data/clean",
        **kwargs
) -> tuple[PriceSeriesDataset, PriceSeriesDataset, PriceSeriesDataset]:
    """
    Load the data for the given symbol and create PriceSeriesDataset objects for it.

    The test set size is determined by the remaining data after the train and
    valid sets are created.

    :param symbol: The symbol to load the data for.
    :param root: The root directory to load the data from.
    :param kwargs: Additional keyword arguments to pass to create_splits.
    :return: A tuple of PriceSeriesDataset objects for the train, valid, and test sets.
    """
    all_features, all_targets = load_symbol(symbol, root=root)
    print(f"Setting up loaders for {symbol} | Features: {all_features.shape}")

    # TODO - Augment feature data here i.e. - financial indicators, one-hot encoding, etc.
    # Question - do we want windowed indicators to cross over splits? Yes, we'd have the date irl

    # Create the splits using the specified sequence length
    train_data, valid_data, test_data = create_splits(
        all_features,
        all_targets,
        **kwargs
    )

    print(f"Split Counts for {symbol} | Train: {len(train_data)} | Valid: {len(valid_data)} | Test: {len(test_data)}")
    return train_data, valid_data, test_data

def run():
    """
    A simple function to show example usage of the dataset utilities.
    """
    # We specify the symbol and other parameters here
    train_dataset, valid_dataset, test_dataset = create_datasets(
        "atnf",
        root="../data/clean",
        seq_len=10,
        train_size=0.75,
        val_size=0.15
    )

    # Once we have the datasets, we can create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # And iterate over the DataLoader objects
    for idx, (x, y) in enumerate(train_loader):
        print(idx, x.shape, y.shape)
        # Print a small slice of each
        if idx == 0:
            print(x[:1], y[:1])


if __name__ == '__main__':
    run()
