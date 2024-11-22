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
from datetime import datetime

import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset

from ._priceseriesdataset import PriceSeriesDataset
from src.indicators import compute_sma, compute_ema, compute_pct_b, compute_macd, compute_momentum, compute_rsi, \
    compute_relative_volume


def print_target_distribution(distributions: list[tuple[str, th.Tensor]]):
    """
    Print the distributions for datasets.

    :param distributions: A list of tuples containing the dataset name and target label counts.
    """
    # Format header
    header = "| Dataset  | 0: Sell | 1: Hold | 2: Buy |"
    separator = "|----------|---------|---------|--------|"
    footer = "|----------|---------|---------|--------|"

    # Format rows
    rows = [
        f"| {name:<8} | {dist[0]:7.2f} | {dist[1]:7.2f} | {dist[2]:6.2f} |"
        for name, dist in distributions
    ]

    print("\n".join([header, separator] + rows + [footer]))


def load_symbol(
        symbol: str,
        root: str = "../data/clean",
        target_type: str = "basic"
) -> tuple[th.Tensor, th.Tensor]:
    """
    Load the data for the given symbol into tensors.

    You may wish to only use a subset of the features, in which case you can
    specify the indices of the features to use.
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
    features = th.tensor(data, dtype=th.float32)
    targets = th.tensor(targets, dtype=th.int64)

    return features, targets


def create_price_series(
        features: th.Tensor,
        targets: th.Tensor,
        seq_len: int = 10,
        close_idx: int = 5,
        price_features: list[int] = None,
        t_0: float = None,
) -> PriceSeriesDataset:
    """
    Create a PriceSeriesDataset object from the given data.

    :return: A PriceSeriesDataset object for the given data.
    """
    # Normalize the price features by the first value of the close price
    features[:, price_features] = features[:, price_features] / features[0, close_idx]

    return PriceSeriesDataset(
        features=features,
        targets=targets,
        price_features=price_features,
        close_idx=close_idx,
        seq_len=seq_len,
        t_0=t_0
    )


def create_splits(
        features: th.Tensor,
        targets: th.Tensor,
        seq_len: int = 10,
        train_start: str = "2020-10-18 00:00:00",
        valid_start: str = "2023-10-18 00:00:00",
        test_start: str = "2024-04-18 00:00:00",
        test_end: str = "2024-10-18 00:00:00",
        price_features: list[int] = None,
        ignore_features: list[int] = None,
        **kwargs
) -> tuple[PriceSeriesDataset, PriceSeriesDataset, PriceSeriesDataset]:
    """
    Split the data into train/valid/test sets and create DataLoader objects.

    :param features: The features to split.
    :param targets: The targets to split.
    :param seq_len: The sequence length of the dataset windows.
    :param train_start: The start date for the training set.
    :param valid_start: The start date for the validation set.
    :param test_start: The start date for the test set.
    :param test_end: The end date for the test set.
    :param price_features: The indices of the price features to normalize.
    :param ignore_features: The indices of the features to ignore.
    :return: A tuple of DataLoader objects for the train, valid, and test sets.
    """
    # Convert dates to timestamps for slicing
    train_start_ts = datetime.fromisoformat(train_start).timestamp()
    valid_start_ts = datetime.fromisoformat(valid_start).timestamp()
    test_start_ts = datetime.fromisoformat(test_start).timestamp()
    test_end_ts = datetime.fromisoformat(test_end).timestamp()

    # Split the targets
    train_mask = (features[:, 0] >= train_start_ts) & (features[:, 0] < valid_start_ts)
    valid_mask = (features[:, 0] >= valid_start_ts) & (features[:, 0] < test_start_ts)
    test_mask = (features[:, 0] >= test_start_ts) & (features[:, 0] <= test_end_ts)

    x_train, y_train = features[train_mask], targets[train_mask]
    x_valid, y_valid = features[valid_mask], targets[valid_mask]
    x_test, y_test = features[test_mask], targets[test_mask]

    t_0 = x_train[0, 0].item()
    price_features = [1, 2, 3, 4, 5] if price_features is None else price_features

    train_set = create_price_series(
        x_train,
        y_train,
        seq_len=seq_len,
        price_features=price_features
    )
    valid_set = create_price_series(
        x_valid,
        y_valid,
        seq_len=seq_len,
        t_0=t_0,
        price_features=price_features
    )
    test_set = create_price_series(
        x_test,
        y_test,
        seq_len=seq_len,
        t_0=t_0,
        price_features=price_features
    )

    print_target_distribution(
        [
            ("Train", train_set.target_dist),
            ("Valid", valid_set.target_dist),
            ("Test", test_set.target_dist)
        ]
    )

    return train_set, valid_set, test_set


def create_datasets(
        symbol: str,
        root: str = "../data/clean",
        fixed_scaling: list[tuple[int, float]] = None,
        **kwargs
) -> tuple[PriceSeriesDataset, PriceSeriesDataset, PriceSeriesDataset]:
    """
    Load the data for the given symbol and create PriceSeriesDataset objects for it.

    The test set size is determined by the remaining data after the train and
    valid sets are created.

    :param symbol: The symbol to load the data for.
    :param root: The root directory to load the data from.
    :param fixed_scaling: A list of tuples of feature index and scaling factor.
    :param kwargs: Additional keyword arguments to pass to create_splits.

    :return: A tuple of PriceSeriesDataset objects for the train, valid, and test sets.
    """
    # Use default time and price features, we'll proxy volume via r_vol
    feature_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9]
    all_features, all_targets = load_symbol(symbol, root=root)
    print(f"Setting up loaders for {symbol} | Features: {all_features.shape}")

    # If scaling passed, perform before slicing
    if fixed_scaling is not None and feature_indices is not None:
        for idx, scale in fixed_scaling:
            all_features[:, idx] = all_features[:, idx] / scale

    adj_close = all_features[:, 5]
    ma_windows = [10, 20, 30]

    # Compute the relative volume before we drop it
    r_vols = compute_relative_volume(all_features[:, 6], windows=ma_windows)  # Volume is index 6

    # Add indicators, we add 11 in total
    sma_features = compute_sma(adj_close, windows=ma_windows)
    ema_features = compute_ema(adj_close, windows=ma_windows)
    pct_b = compute_pct_b(adj_close)
    macd_h = compute_macd(adj_close)
    momentum = compute_momentum(adj_close)
    rsi = compute_rsi(adj_close)

    # Count the default features without volume
    all_features = all_features[:, feature_indices]
    n_features = all_features.shape[1]
    all_features = th.cat([
        all_features,
        sma_features,
        ema_features,
        pct_b,
        macd_h,
        momentum,
        rsi,
        r_vols
    ], dim=1)

    # 1:5 are open/high/low/close, adj_close so we add them and the moving averages
    price_features = [i + 1 for i in range(5)] + [n_features + i for i in range(6)]
    kwargs["price_features"] = price_features

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
        fixed_scaling=[(7, 3000.), (8, 12.), (9, 31.)],
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
