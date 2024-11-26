import os

import numpy as np
import pandas as pd




def gen_date_dataset(
        start_date: str = "1900-01-01",
        end_date: str = "2100-01-01",
        file_path: str = "data/date_pretraining.csv"
) -> np.array:
    """
    Generate a range of dates and convert to yyyy, mm, dd then drop to disk.

    If the file already exists, read from disk and return the data. Otherwise,
    we generate a range of dates using pandas functionality to ease the process.
    The dates are then stored in column format with year, month, and day. The
    year, month and day are normalized to the range [0, 1] by using the following
    scaling factors:
    - year: 3000
    - month: 12
    - day: 31

    :param start_date: The start date for the range of dates.
    :param end_date: The end date for the range of dates.
    :param file_path: The file path to store the data.
    :return: A numpy array with year, month, and day normalized values.
    """
    if os.path.exists(file_path):
        dates_df = pd.read_csv(file_path)
        return dates_df.values

    dates = pd.date_range(start=start_date, end=end_date)
    dates_df = pd.DataFrame(dates, columns=["date"])
    dates_df["year"] = dates_df["date"].dt.year / 3000.0
    dates_df["month"] = dates_df["date"].dt.month / 12.0
    dates_df["day"] = dates_df["date"].dt.day / 31.0

    # Day of the week
    # dates_df["dow"] = dates_df["date"].dt.dayofweek / 6.0
    dates_df.drop(columns=["date"], inplace=True)
    dates_df.to_csv("data/date_pretraining.csv", index=False)

    return dates_df.values


def run_pretraining():
    date_range = gen_date_dataset()
    print(date_range[0:5])

if __name__ == '__main__':
    run_pretraining()
