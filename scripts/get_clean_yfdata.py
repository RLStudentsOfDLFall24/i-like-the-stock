import argparse

import yfinance as yf


def get_yfdata():
    # Define the CLI arguments
    parser = argparse.ArgumentParser(description="Download historical stock data using yfinance.")
    parser.add_argument("--ticker", type=str, default="^SPX", help="Ticker symbol (default: ^GSPC for SPX)")
    parser.add_argument("--start", type=str, default="2018-10-18", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, default="2024-10-31", help="End date in YYYY-MM-DD format")

    args = parser.parse_args()

    # Extract the arguments
    ticker = args.ticker
    start_date = args.start
    end_date = args.end

    # Download the historical data and put it in the same format as our other .dat files
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    updated_cols = [col[0] for col in data.columns]
    data.columns = updated_cols
    data = data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

    data.index = data.index.map(lambda x: x.strftime("%b %d, %Y"))


    # Format the filename
    filename = f"yf_data/{ticker.replace('^', '').lower()}.dat"
    data.to_csv(filename, sep="\t")


if __name__ == '__main__':
    get_yfdata()
