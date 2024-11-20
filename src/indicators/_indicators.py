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
        smas[:w - 1, ix] = cum_prices[:w - 1] / th.arange(1, w)

    # # Normalize the SMA values by the initial price
    # smas = smas / prices[0]

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

    # # Normalize the EMA values by the initial price
    # emas = emas / prices[0]

    return emas


def compute_macd(
        prices: th.Tensor,
        slow: int = 26,
        fast: int = 12,
        signal: int = 9
) -> th.Tensor:
    """
    Compute the moving average convergence divergence (MACD) histogram.

    The MACD is computed as the difference between the slow and fast moving averages.
    We then take the difference between the MACD and the signal line, which gives
    us the MACD histogram.

    :param prices: The prices to compute the MACD for
    :param slow: The slow moving average window, default is 26
    :param fast: The fast moving average window, default is 12
    :param signal: The signal line window, default is 9
    :return: A tensor of MACD histogram values T x 1
    """
    # Get the 12, 26 day smas and the signal EMA
    smas = compute_sma(prices, [slow, fast])
    signal_line = compute_ema(prices, [signal])

    # Compute the histogram values
    macd_h = smas[:, 1] - smas[:, 0] - signal_line[:, 0]
    return macd_h.unsqueeze(1)


def compute_pct_b(prices: th.Tensor, sma_size: int = 20) -> th.Tensor:
    """
    Compute the percentage B indicator for a given set of prices.

    :param prices: The prices to compute the indicator for
    :param sma_size: The simple moving average window size, default is 20
    :return: A tensor of percentage B values T x 1
    """
    assert sma_size > 1, "SMA size must be greater than 1"
    assert prices.shape[0] > sma_size, "Prices must have more data than the SMA size"

    # Compute the sma
    sma_20 = compute_sma(prices, [sma_size])[:, 0]

    # Grab the sd, but we'll only be defined from the sma_size - 1
    sd = th.full_like(prices, fill_value=th.nan)
    sd[sma_size - 1:] = prices.unfold(0, sma_size, 1).std(1)

    # Upper and Lower Bands are sma +/- the sd
    upper_band = sma_20 + 2 * sd
    lower_band = sma_20 - 2 * sd

    # We compute %B similar to min-max scaling, subtracting the lower band
    return ((prices - lower_band) / (upper_band - lower_band)).unsqueeze(1)

def compute_momentum(prices: th.Tensor, window: int = 10) -> th.Tensor:
    """
    Compute the momentum indicator for a given set of prices.

    :param prices: The prices to compute the indicator for
    :param window: The window size for the momentum indicator
    :return: A tensor of momentum values T x 1
    """
    assert window > 1, "Window size must be greater than 1"
    assert prices.shape[0] > window, "Prices must have more data than the window size"

    # Compute momentum
    normed_prices = prices / prices[0]

    momentum = th.full_like(normed_prices, fill_value=th.nan)
    momentum[window:] = (normed_prices[window:] / normed_prices[:-window]) - 1
    return momentum.unsqueeze(1)

def compute_rsi(prices: th.Tensor, window: int = 14) -> th.Tensor:

    result = th.full_like(prices, fill_value=th.nan).unsqueeze(1)
    # Compute the daily differences
    deltas = prices[1:] - prices[:-1]

    # We can clip up and down days
    gain_days, loss_days = deltas.clone().clip(min=0), deltas.clone().clip(max=0).abs()
    # gain_days = gain_days.clip(min=0)
    # loss_days =loss_days.clip(max=0)

    # Compute n-day EMA for gains and losses
    ema_gain = compute_ema(gain_days, [window])
    ema_loss = compute_ema(loss_days, [window])

    # Finally compute the RSI
    result[1:] = 1 - (1/ (1 + (ema_gain / ema_loss)))
    # We need to cap at 100
    result.clip(max=1)
    return result


def run_example():
    # Run a simple example to test the functions
    # a random tensor of prices for 100 days, all strictly between 0 and 100
    th.manual_seed(42)
    prices = th.rand(100)

    windows = [10, 20, 30]
    ema = compute_ema(prices, windows)
    sma = compute_sma(prices, windows)
    pct_b = compute_pct_b(prices, sma_size=3)
    macd = compute_macd(prices)
    moment = compute_momentum(prices, window=10)
    rsi = compute_rsi(prices, window=14)
    # Make sure we can concat all of these features

    features = th.cat([prices.unsqueeze(1), ema, sma, pct_b, macd, moment, rsi], dim=-1)
    print(features)


if __name__ == '__main__':
    run_example()
