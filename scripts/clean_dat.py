from datetime import datetime

import numpy as np
import pandas as pd


def generate_target_labels(prices: np.array, bound: float = 0.005) -> np.array:
    """
    Generate target labels for a given dataset.

        1. Compute the next day return for p_t and p_{t+1}
        2. Assign a label { -1: Sell, 0: Hold, 1: Buy } based on where the return falls
            - We use the bound to determine the label

    :param prices: The data we need to generate labels for
    :param bound: The threshold for determining the label
    :return: A numpy array of labels
    """
    daily_rets = (prices[1:] - prices[:-1]) / prices[:-1]
    labels = np.full_like(prices, fill_value=0, dtype=np.int8)
    labels[:-1][daily_rets >= bound] = 1
    labels[:-1][daily_rets <= -bound] = -1

    return labels


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


def process_dat_files(n_cols: int = 7):
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
            cleaned["target"] = generate_target_labels(cleaned["adj_close"].values)

            with open(f"../data/clean/{symbol}.csv", "w", newline='') as f_out:
                cleaned.to_csv(f_out, index=False)
            print(f"Cleaned {symbol}.dat | Processed {len(cleaned)} rows | Saved to {symbol}.csv\n")

    except Exception as e:
        print(f"Error processing dat files: {e}")
        raise


def run():
    process_dat_files()


if __name__ == '__main__':
    run()
