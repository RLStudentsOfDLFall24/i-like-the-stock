import argparse
import csv
from datetime import datetime


def process_dat_files(n_cols: int = 7):
    symbols = ["atnf", "biaf", "bivi", "cycc", "vtak"]
    try:
        for symbol in symbols:
            print(f"{"Processing " + symbol + ".dat ":#^50}")
            with open(f"../data/raw/{symbol}.dat", "r") as f:
                cleaned = []
                header_processed = False
                for line in f:
                    # Skip the original header line, we'll write a new one
                    if not header_processed:
                        header_processed = True
                        continue

                    columns = [col.strip() for col in line.strip().split('\t')]
                    if len(columns) == n_cols:
                        date_obj = datetime.strptime(columns[0], "%b %d, %Y")
                        # Standardize the date string format
                        columns[0] = date_obj.strftime('%Y-%m-%dT%H:%M:%SZ')
                        columns.insert(1, int(date_obj.timestamp()))

                        # Clean the volume (last) column to be either None or a float
                        if columns[-1] == "-":
                            columns[-1] = 0
                        else:
                            columns[-1] = float(columns[-1].replace(",", ""))

                        cleaned.append(columns)
                    else:
                        print(f"Skipping line: {line}")

            # Convert to numpy and save as csv
            with open(f"../data/clean/{symbol}.csv", "w", newline='') as f_out:
                writer = csv.writer(f_out)
                writer.writerow([
                    "utcdate", "timestamp", "open", "high", "low", "close", "adj_close", "volume"
                ])

                for row in cleaned:
                    writer.writerow(row)
            print(f"Cleaned {symbol}.dat | Processed {len(cleaned)} rows | Saved to {symbol}.csv\n")

    except Exception as e:
        print(f"Error processing dat files: {e}")
        raise


def run():
    process_dat_files()


if __name__ == '__main__':
    run()
