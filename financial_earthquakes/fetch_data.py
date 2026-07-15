"""
Fetch hourly OHLC data for a ticker into the CSV format this project expects
(the 3-header-row yfinance layout, identical to AA_h.csv).

Run wherever outbound market-data access exists (e.g. your Colab):

    pip install yfinance
    python3 fetch_data.py BMNR          ->  BMNR_h.csv

then place the file next to run_analysis.py and run:

    python3 run_analysis.py BMNR_h.csv
    python3 run_daily_walkforward.py BMNR_h.csv

NOTE: the managed environment this project was built in blocks market-data
hosts (Yahoo etc.), so this script is for YOUR machine/Colab; the analysis
itself runs anywhere.
"""

import sys

import yfinance as yf


def fetch(ticker: str, period: str = "730d") -> str:
    df = yf.download(ticker, interval="1h", period=period,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"no data returned for {ticker}")
    out = f"{ticker.upper()}_h.csv"
    df.to_csv(out)
    print(f"{out}: {len(df)} hourly bars, {df.index[0]} -> {df.index[-1]}")
    return out


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else "BMNR")
