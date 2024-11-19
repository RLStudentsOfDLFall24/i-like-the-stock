"""
Indicators for averages on torch tensors - e.g. SMA, EMA
"""

import torch as th


def compute_sma(prices: th.Tensor, windows: list[int] = None) -> th.Tensor:
    """
    Compute the simple moving averages for a given set of prices.

    Dates without enough data will be filled with NaN. Simple moving averages
    are window based average over a sequence of prices. The windows will be
    computed for each window size in the list.

    :param prices: A tensor of T prices T
    :param windows: A list of W window sizes to compute the averages for
    :return: A tensor of simple moving averages T x W
    """
    windows = windows if windows is not None else [5, 10, 20]
    smas = th.full(size=(prices.shape[0], len(windows)), fill_value=th.nan)

    cum_prices = prices.cumsum(dim=0)

    for ix, w in enumerate(windows):
        smas[w - 1:, ix] = prices.unfold(0, w, 1).mean(1)

        # Use a rolling average over the nan values
        smas[:w - 1, ix] =  cum_prices[:w - 1]/ th.arange(1, w)

    # Normalize the SMA values by the initial price
    smas = smas / prices[0]

    return smas


def compute_ema(prices: th.Tensor, windows: list[int] = None) -> th.Tensor:
    """
    Compute the exponential moving averages for a given set of prices.

    Unlike SMA, the values for EMA can start at the beginning of the time series.
    Exponential moving averages are weighted window based average over a sequence
    of prices. The windows will be computed for each window size in the list.

    :param prices: A tensor of prices T x 1
    :param windows: A list of W window sizes to compute the averages for
    :return: A tensor of exponential moving averages T x W
    """
    windows = th.tensor(windows) if windows is not None else th.tensor([5, 10, 20])

    emas = th.zeros(prices.shape[0], len(windows))
    alphas = 2 / (windows + 1)

    # Set the initial price and iterate
    emas[0] = prices[0]
    for t in range(1, prices.shape[0]):
        emas[t] = alphas * prices[t] + (1 - alphas) * emas[t - 1]

    # Normalize the EMA values by the initial price
    emas = emas / prices[0]

    return emas


def run_example():
    # Run a simple example to test the functions
    prices = th.tensor([1.4, 1.2, 1.4, 1.6, 1.8, 2.0])
    windows = [2, 3, 4]
    ema = compute_ema(prices, windows)
    print(ema)

    sma = compute_sma(prices, windows)
    print(sma)
    pass


if __name__ == '__main__':
    run_example()
