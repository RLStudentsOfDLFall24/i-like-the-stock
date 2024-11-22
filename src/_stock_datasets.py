# """
# @Author: Sakae Watanabe
# @Date: 2024-11-14
# @Description: Utilities for loading and processing of train/test data.
#
#
# We export the following functions and classes via __init__.py:
#     - PriceSeriesDataset
#     - create_datasets
#
# Example importing from top level:
#     from src import create_datasets, PriceSeriesDataset
# """
# import numpy as np
# import torch as th
# from torch.utils.data import DataLoader, Dataset
#
# from src.indicators import compute_sma, compute_ema
#
#
# class PriceSeriesDataset(Dataset):
#     """
#     A simple Dataset class for loading our using the torch APIs
#
#     For simplicity, expects that the features are already segmented into windows
#     of the desired sequence length. This could be adjusted in the future to allow
#     the sequence length to be a parameter of the SequenceDataset class itself
#     to abstract away the windowing process.
#     """
#
#     features: th.Tensor
#     """The features of the dataset"""
#
#     targets: th.Tensor
#     """The targets of the dataset"""
#
#     seq_len: int
#     """The sequence length of the dataset windows"""
#
#     feature_dim: int
#     """The size of a single feature vector in the dataset"""
#
#     t_0: float
#     """The first time value in the dataset, a UNIX timestamp"""
#
#     time_idx: th.Tensor
#     """The time index of the dataset"""
#
#     def __init__(
#             self,
#             features: th.Tensor,
#             targets: th.Tensor,
#             close_idx: int = 5,
#             seq_len: int = 10,
#             t_0: float = None,
#             price_features: list[int] = None,
#     ):
#         """
#         Initialize the dataset with the given features and targets.
#
#         :param features: The time series features for the dataset.
#         :param targets: The targets for the dataset at each time step.
#         :param close_idx: The index of the close price column.
#         :param seq_len: The sequence length of the dataset windows.
#         :param t_0: The first time value in the dataset.
#         :param price_features: The indices of the price features to normalize.
#         """
#         # Seq len can't be greater than the number of features
#         assert seq_len <= features.shape[0], "Sequence length must be less than the number of features"
#         assert seq_len > 0, "Sequence length must be greater than 0"
#
#         # We normalize price values by the first value of adj_close (col 5)
#         price_features = price_features if price_features is not None else [1, 2, 3, 4, 5]
#         features[:, price_features] = features[:, price_features] / features[0, close_idx]
#
#         # Don't use the last window as it will have no target
#         self.time_idx = features[:, 0].clone()
#         self.t_0 = t_0 if t_0 is not None else self.time_idx[0].item()
#
#         # # We subtract the first time value to get a relative time index starting at 0
#         self.features = self.__create_seq_windows(features, seq_len, self.t_0)
#
#         self.targets = targets[seq_len:]
#         self.feature_dim = self.features.shape[-1]
#
#     @staticmethod
#     def __create_seq_windows(
#             features: th.Tensor,
#             seq_len: int,
#             time_offset: float
#     ) -> th.Tensor:
#         """
#         Subtracts the first time value to get a relative time index.
#
#         :param features: The features to create windows for.
#         :param seq_len: The sequence length of the windows.
#         :param time_offset: The first time value in the dataset.
#         :return: A tensor of windows of the given sequence
#         """
#         features[:, 0] = (features[:, 0] - time_offset) / 86400  # seconds in a day
#         return features.unfold(0, seq_len, 1)[:-1].transpose(1, 2)
#
#     def __len__(self):
#         return self.features.shape[0]
#
#     def __getitem__(self, idx):
#         return self.features[idx], self.targets[idx]
#
#
# def load_symbol(
#         symbol: str,
#         root: str = "../data/clean",
#         target_type: str = "basic"
# ) -> tuple[th.Tensor, th.Tensor]:
#     """
#     Load the data for the given symbol into tensors.
#
#     You may wish to only use a subset of the features, in which case you can
#     specify the indices of the features to use.
#     :param symbol: The symbol to load the data for.
#     :param root: The root directory to load the data from.
#     :param target_type: The type of target to load, default "basic".
#     :return: A tuple of torch.Tensors containing the features and targets
#     """
#     data = np.genfromtxt(
#         f"{root}/{symbol}.csv",
#         delimiter=",",
#         skip_header=1,
#     )
#
#     targets = np.genfromtxt(
#         f"{root}/{symbol}_target_{target_type}.csv",
#         dtype=int,
#         delimiter=","
#     )
#     # Split the features and targets
#     features = th.tensor(data, dtype=th.float32)
#     targets = th.tensor(targets, dtype=th.int64)
#
#     return features, targets
#
#
# def create_splits(
#         features: th.Tensor,
#         targets: th.Tensor,
#         train_size: float = 0.75,
#         val_size: float = 0.15,
#         seq_len: int = 10,
#         **kwargs
# ) -> tuple[PriceSeriesDataset, PriceSeriesDataset, PriceSeriesDataset]:
#     """
#     Split the data into train/valid/test sets and create DataLoader objects.
#
#     :param features: The features to split.
#     :param targets: The targets to split.
#     :param train_size: The size of the training set.
#     :param val_size: The size of the validation set.
#     :param seq_len: The sequence length of the dataset windows.
#     :return: A tuple of DataLoader objects for the train, valid, and test sets.
#     """
#     # Get counts for splits
#     n_rows = features.shape[0]
#     n_train, n_val = int(n_rows * train_size), int(n_rows * val_size)
#     n_test = n_rows - n_train - n_val
#
#     x_train, x_valid, x_test = th.split(features, [n_train, n_val, n_test])
#     y_train, y_valid, y_test = th.split(targets, [n_train, n_val, n_test])
#
#     # Create the datasets
#     t_0 = x_train[0, 0].item()
#
#     return (
#         PriceSeriesDataset(x_train, y_train, seq_len=seq_len),
#         PriceSeriesDataset(x_valid, y_valid, seq_len=seq_len, t_0=t_0),
#         PriceSeriesDataset(x_test, y_test, seq_len=seq_len, t_0=t_0)
#     )
#
#
# def create_datasets(
#         symbol: str,
#         root: str = "../data/clean",
#         fixed_scaling: list[tuple[int, float]] = None,
#         feature_indices: list[int] = None,
#         add_windowed_avgs: bool = False,
#         **kwargs
# ) -> tuple[PriceSeriesDataset, PriceSeriesDataset, PriceSeriesDataset]:
#     """
#     Load the data for the given symbol and create PriceSeriesDataset objects for it.
#
#     The test set size is determined by the remaining data after the train and
#     valid sets are created.
#
#     :param symbol: The symbol to load the data for.
#     :param root: The root directory to load the data from.
#     :param fixed_scaling: A list of tuples of feature index and scaling factor.
#     :param feature_indices: The indices of the features to use.
#     :param add_windowed_avgs: Whether to add windowed averages to the features.
#     :param kwargs: Additional keyword arguments to pass to create_splits.
#
#     :return: A tuple of PriceSeriesDataset objects for the train, valid, and test sets.
#     """
#     all_features, all_targets = load_symbol(symbol, root=root)
#     print(f"Setting up loaders for {symbol} | Features: {all_features.shape}")
#
#     # If scaling passed, perform before slicing
#     if fixed_scaling is not None and feature_indices is not None:
#         for idx, scale in fixed_scaling:
#             all_features[:, idx] = all_features[:, idx] / scale
#
#     if feature_indices is None:
#         # Use the default feature indices, the first 7 columns
#         feature_indices = [0, 1, 2, 3, 4, 5, 6] # Todo set back to default
#
#     all_features = all_features[:, feature_indices]
#
#     # TODO - Augment feature data here i.e. - financial indicators, one-hot encoding, etc.
#     # Add simple moving averages
#     if add_windowed_avgs:
#         adj_close = all_features[:, 5]
#         windows = [5, 10, 20]
#         sma_features = compute_sma(adj_close, windows=windows)
#         ema_features = compute_ema(adj_close, windows=windows)
#         all_features = th.cat([all_features, sma_features, ema_features], dim=1)
#
#     # Drop the volume column, too much noise in column 6
#     all_features = th.cat([all_features[:, :6], all_features[:, 7:]], dim=1)
#
#     # Create the splits using the specified sequence length
#     train_data, valid_data, test_data = create_splits(
#         all_features,
#         all_targets,
#         **kwargs
#     )
#
#     # TODO - move this information to where we actually create the datasets with sequences
#     print(f"Split Counts for {symbol} | Train: {len(train_data)} | Valid: {len(valid_data)} | Test: {len(test_data)}")
#     return train_data, valid_data, test_data
#
#
# def run():
#     """
#     A simple function to show example usage of the dataset utilities.
#     """
#     # We specify the symbol and other parameters here
#     train_dataset, valid_dataset, test_dataset = create_datasets(
#         "atnf",
#         root="../data/clean",
#         seq_len=10,
#         train_size=0.75,
#         val_size=0.15,
#         fixed_scaling = [(7, 3000.), (8, 12.), (9, 31.)],
#         feature_indices=[i for i in range(10)],
#         add_windowed_avgs=True
#     )
#
#     # Once we have the datasets, we can create DataLoader objects
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     # valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
#     # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#     # Time indices [0, 7, 8, 9], maybe we drop volume?
#
#     # And iterate over the DataLoader objects
#     for idx, (x, y) in enumerate(train_loader):
#         print(idx, x.shape, y.shape)
#         # Print a small slice of each
#         if idx == 0:
#             print(x[:1], y[:1])
#
#
# if __name__ == '__main__':
#     run()
