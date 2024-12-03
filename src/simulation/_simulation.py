import numpy as np
import pandas as pd


def simulate_trades(
        prices: np.array,
        trades: np.array,
        timestamps: np.array,
        cash: float = 100_000,
        commission: float = 10.00
) -> pd.DataFrame:
    """
    Simulate the given trades with the starting cash balance.

    :param prices: The historical prices for the asset being simulated
    :param trades: The list of trades to make for the given asset
    :param timestamps: The index of timestamps related to these trades
    :param cash: The starting balance before any trades are executed
    :param commission: The assumed commission for the trades being executed.
            The commission fee applies to both buys/sells
    :return: A dataframe with historical value of the trades and cash.
    """
    # assert prices.shape == trades.shape
    # TODO create a dataframe index from the timestamps
    # TODO put the prices, trades, cash in the dataframe
    # TODO iterate through the dates to compute the ending value
    pass


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
    valid_indices_from_seq =( valid.features[:, -1, 0].detach().numpy() * 86400) + valid.t_0
    are_close = np.allclose(valid_indices_from_seq, valid.time_idx.detach().numpy())
    print(f"Close? {are_close}")

if __name__ == '__main__':
    run()
