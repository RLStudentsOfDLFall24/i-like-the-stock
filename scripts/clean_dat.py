import argparse
import csv
from datetime import datetime


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Process .dat file and add a unix timestamp")
    parser.add_argument("--symbol", type=str, default="atnf", help="Symbol to clean raw input")
    return vars(parser.parse_args())


def process_dat(in_file: str, n_cols: int = 7):
    try:
        cleaned = []
        header_processed = False
        with open(f"../data/raw/{in_file}.dat", "r") as f:
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
        with open(f"../data/clean/{in_file}.csv", "w", newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow([
                "utcdate", "timestamp", "open", "high", "low", "close", "adj_close", "volume"
            ])

            for row in cleaned:
                writer.writerow(row)
        print(f"Cleaned {in_file}.dat | Processed {len(cleaned)} rows | Saved to {in_file}.csv")

    except FileNotFoundError:
        print(f"File not found: {in_file}")
        raise


def run():
    args = parse_args()
    process_dat(args["symbol"])


if __name__ == '__main__':
    run()
