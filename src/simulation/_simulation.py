import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_trades(
        prices: np.array,
        trades: np.array,
        timestamps: np.array,
        cash: float = 10_000,
        commission: float = 10.00,
        shares_per_trade: int = 1_000
) -> pd.DataFrame:
    """
    Simulate the given trades with the starting cash balance.

    :param prices: The historical prices for the asset being simulated
    :param trades: The list of trades to make for the given asset
    :param timestamps: The index of timestamps related to these trades
    :param cash: The starting balance before any trades are executed
    :param commission: The assumed commission for the trades being executed.
            The commission fee applies to both buys/sells
    :param shares_per_trade: The number of shares to buy/sell per trade
    :return: A dataframe with historical value of the trades and cash.
    """
    # Convert the time stamps to a pandas datetime index
    timestamps = pd.to_datetime(timestamps, unit='s')

    # Setup prices for the stock and value of cash
    price_ix = np.ones((prices.shape[0], 2))
    price_ix[:, 0] = prices

    # Compute commissions and trades 0: Stock, 1: Cash
    trade_vals = np.zeros_like(price_ix)
    trade_vals[:, 0] = trades * shares_per_trade
    commissions = np.zeros_like(trades)
    commissions[trades != 0] = commission

    # Compute cash impact of each trade
    trade_vals[:, 1] = -(trade_vals[:, 0] * price_ix[:, 0]) - commissions

    # Iterate through the trades and compute the ending value
    holding_vals = np.zeros_like(trade_vals)
    holding_vals[0, 1] = cash  # Set the initial cash value
    for t in range(1, trade_vals.shape[0]):
        holding_vals[t] = holding_vals[t - 1, :] + trade_vals[t - 1, :]

    adjusted_val = (holding_vals * price_ix).sum(axis=1)
    adjusted_val /= adjusted_val[0]
    adjusted_val = np.round(adjusted_val, 2)

    result_df = pd.DataFrame({
        "value": adjusted_val,  # The total value of the portfolio
        "price": np.round(prices / prices[0], 2),  # The normalized price
    }, index=timestamps)

    return result_df


def get_long_short_trades(actions: np.array) -> np.ndarray:
    """
    Convert a list of actions to a long-short strategy.

    Actions must be one of 0: Sell, 1: Hold, 2: Buy. Repeated actions are ignored
    and converted to non-operation. For the long-short strategy we'll always
    reverse our position if the action opposes our current position.
        e.g. - We begin with a sell -1 and receive a signal to buy. We double
            buy to take a net position of +1

    Actions are converted in the following manner:
        - 0 Sell: Sell actions are converted to a -1 for a single sell, -2 for a double sell
        - 1 Hold: Converted to a 0 to represent no trade
        - 2 Buy: Converted to a 1 for a single buy, 2 for a double buy

    :param actions: An array of actions in the set {0: Sell, 1: Hold, 2: Buy}
    :return: An iterable of trades created from the supplied actions.

    Examples:
    >>> get_long_short_trades(np.array([2, 1, 0, 2, 0, 0, 2]))
    array([ 1.,  0., -2.,  2., -2.,  0.,  2.])

    >>> get_long_short_trades(np.array([1, 1, 1, 1, 1, 1, 1]))
    array([0., 0., 0., 0., 0., 0., 0.])

    >>> get_long_short_trades(np.array([2, 2, 2]))
    array([1., 0., 0.])

    >>> get_long_short_trades(np.array([0, 0, 0]))
    array([-1.,  0.,  0.])

    >>> get_long_short_trades(np.array([1, 1, 1]))
    array([0., 0., 0.])

    >>> get_long_short_trades(np.array([0, 1, 1, 2]))
    array([-1.,  0.,  0.,  2.])
    """
    # Initialize the trades
    trades = np.zeros(len(actions))
    actionable = actions != 1

    # If we don't have any trades we can stop now
    if not np.any(actionable):
        return trades

    # Set the action mappings
    trades[actions == 2] = 1
    trades[actions == 0] = -1
    first_trade = np.where(actionable)[0][0]

    # We have trades, we need to check for repeat actions against the next day
    repeated_action = np.full(len(actions), fill_value=False)
    next_day = np.zeros_like(actions[actionable])
    next_day[1:] = actions[actionable][:-1]

    repeated_action[actionable] = actions[actionable] == next_day
    repeated_action[first_trade] = False  # We don't ignore the first trade
    trades[repeated_action] = 0

    trades[first_trade + 1:] *= 2
    return trades


def run():
    """A simple run script to test functionality"""
    time_steps = np.array([0, 1, 2, 3, 4])
    trades_t = np.zeros_like(time_steps)
    actions = np.array([0, 1, 1, 2])
    trades_t[:-1] = get_long_short_trades(actions)
    print(trades_t)

    # Put together some tests for the trading
    from src.dataset import create_datasets

    train, valid, test = create_datasets(
        "atnf",
        root='../../data/clean',
        seq_len=10,
    )

    # Compare against the stored time_idx for sanity
    valid_indices_from_seq = (valid.features[:, -1, 0].detach().numpy() * 86400) + valid.t_0
    are_close = np.allclose(valid_indices_from_seq, valid.time_idx.detach().numpy())
    print(f"Close? {are_close}")

    # Test the trade simulation
    dummy_trades = np.zeros_like(valid.time_idx)
    dummy_trades[0] = 1  # We just buy on the first day and that's it
    dummy_trades[10] = -2
    results = simulate_trades(
        valid.unscaled_prices.detach().numpy(),
        dummy_trades,
        valid.time_idx.detach().numpy()
    )

    # Simple example of plotting the results, we could concatenate multiple symbols
    # and/or use seaborn instead
    plot = results.plot(
        y=["value", "price"],
        title="Portfolio Value and Normalized Price",
    )
    plot.set_xlim([results.index[0], results.index[-1]])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    run()
