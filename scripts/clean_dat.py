import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import yaml


def create_basic_targets(prices: np.array, bound: float = 0.005, **kwargs) -> np.array:
    """
    Create basic target labels for a given dataset based on the daily returns.

        1. Compute the next day return for p_t and p_{t+1}
        2. Assign a label { 0: Sell, 1: Hold, 2: Buy } based on where the return falls
            - We use the bound to determine the label

    :param prices: The data we need to generate labels for
    :param bound: The threshold for determining the label
    :param kwargs: Additional arguments for the target generation
    :return: A numpy array of labels
    """
    daily_rets = (prices[1:] - prices[:-1]) / prices[:-1]
    labels = np.full_like(prices, fill_value=1, dtype=np.int8)
    labels[:-1][daily_rets >= bound] = 2
    labels[:-1][daily_rets <= -bound] = 0
    return labels


def generate_target_labels(prices: np.array, target_type: str = "basic", **kwargs) -> np.array:
    """
    Generate target labels for a given dataset.

    :param prices: The data we need to generate labels for
    :param target_type: The type of target labels to generate
    :param kwargs: Additional arguments for the target generation
    :return: A numpy array of one-hot encoded labels
    """
    match target_type:
        case "basic":
            labels = create_basic_targets(prices, **kwargs)
        case _:
            raise ValueError(f"Unknown target type: {target_type}")

    assert labels is not None, "No labels were generated"
    assert len(prices) == len(labels), "Length of prices and labels must match"

    # Log the distribution of the labels
    dist = np.unique(labels, return_counts=True)[1] / labels.shape[0]
    print(f"Labels: Sell {dist[0]:.2f} | Hold {dist[1]:.2f} | Buy {dist[2]:.2f}")

    # One-hot encode the labels and return
    return np.eye(3, dtype=int)[labels]


def read_dat_file(symbol: str, n_cols: int) -> pd.DataFrame:
    """
    Read a .dat file and clean it up for processing.

    :param symbol: The symbol of the stock we're processing
    :param n_cols: The number of columns in the file
    :return: A DataFrame of the cleaned data
    """
    print(f"{"Processing " + symbol + ".dat ":#^50}")
    with open(f"../data/raw/{symbol}.dat", "r") as f:
        cleaned = []
        header_processed = False
        for line in f:
            # Skip the original header line, we'll write a new one
            if not header_processed:
                header_processed = True
                continue

            columns = [
                col.strip().replace(",", "")
                for col in line.strip().split('\t')
            ]

            if len(columns) == n_cols:
                date_obj = datetime.strptime(columns[0], "%b %d %Y")
                # Add a unix timestamp
                columns[0] = date_obj.strftime('%Y-%m-%dT%H:%M:%S')
                columns.insert(1, int(date_obj.timestamp()))

                if columns[-1] == "-":
                    columns[-1] = 0
                else:
                    columns[-1] = float(columns[-1])

                cleaned.append(tuple(columns))
            else:
                print(f"Skipping line: {line}")

    # Sort by timestamp and return a DataFrame
    cleaned.sort(key=lambda x: x[0])
    df = pd.DataFrame(
        cleaned,
        columns=[
            "utc_date", "timestamp", "open", "high", "low", "close",
            "adj_close", "volume"
        ]
    )

    df["utc_date"] = pd.to_datetime(df["utc_date"])
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df["volume"] = df["volume"].astype(np.int64)
    for col in ["open", "high", "low", "close", "adj_close"]:
        df[col] = df[col].astype(np.float64)

    return df


def process_dat_files(n_cols: int, target_type: str, **kwargs) -> None:
    """
    Process all the .dat files in the raw directory and save them to the clean directory.

    Run some basic cleaning on the data and generate target labels.

    :param n_cols: The number of columns in the .dat file
    :param target_type: The type of target labels to generate
    :param kwargs: Additional arguments for the target generation
    """
    symbols = [
        "atnf",
        "biaf",
        "bivi",
        "cycc",
        "vtak"
    ]

    try:
        for symbol in symbols:
            cleaned = read_dat_file(symbol, n_cols)
            # Produce the target labels using adj_close
            target_labels = generate_target_labels(
                cleaned["adj_close"].values,
                target_type=target_type,
                **kwargs
            )

            with open(f"../data/clean/{symbol}.csv", "w", newline='') as data_out:
                cleaned.to_csv(data_out, index=False)
            print(f"Cleaned {symbol}.dat | Processed {len(cleaned)} rows | Saved to {symbol}.csv\n")

            with open(f"../data/clean/{symbol}_target_{target_type}.csv", "w", newline='') as target_out:
                np.savetxt(target_out, target_labels, delimiter=",", fmt="%d")

    except Exception as e:
        print(f"Error processing dat files: {e}")
        raise


def run():
    parser = argparse.ArgumentParser(description="Clean .dat files and generate target labels")
    parser.add_argument("--config", type=str, default="default",  help="The name of the configuration file to use")
    args = parser.parse_args()

    with open(f"{args.config}.yml", "r") as f:
        config = yaml.safe_load(f)

    print("Loaded Configuration:")
    print(config)

    process_dat_files(**config)
    print("Processing complete")


if __name__ == '__main__':
    run()
