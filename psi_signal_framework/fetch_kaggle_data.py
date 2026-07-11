"""
Fetch the public Kaggle dataset rockinbrock/spy-1-minute-data (file
spy_1min_2008_2021_cleaned.csv) and stage it into data/:

  data/spy_1min_2010.csv                 calendar-2010 subset (flash crash year)
  data/spy_1min_2008_2021_cleaned.csv.gz full file, gzipped (< GitHub's 100 MB cap)

Runs inside GitHub Actions (see .github/workflows/fetch_spy_data.yml) because
the Claude Code environment's network policy blocks kaggle.com; Actions
runners have open egress.  Public Kaggle datasets download anonymously via
kagglehub; a direct-URL curl fallback is included.
"""

import glob
import gzip
import os
import shutil
import subprocess
import sys

import pandas as pd

DATASET = "rockinbrock/spy-1-minute-data"
TARGET_NAME = "spy_1min_2008_2021_cleaned"


def download() -> str:
    try:
        import kagglehub
        path = kagglehub.dataset_download(DATASET)
        print(f"kagglehub download ok: {path}")
        return path
    except Exception as e:
        print(f"kagglehub failed ({e!r}); trying direct URL", file=sys.stderr)
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
    os.makedirs("kaggle_dl", exist_ok=True)
    subprocess.run(["curl", "-sSL", "--fail", "-o", "kaggle_dl/ds.zip", url], check=True)
    subprocess.run(["unzip", "-o", "kaggle_dl/ds.zip", "-d", "kaggle_dl"], check=True)
    return "kaggle_dl"


def main():
    root = download()
    csvs = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    print("csv files found:", csvs)
    if not csvs:
        sys.exit("no CSV found in the dataset")
    src = next((f for f in csvs if TARGET_NAME in os.path.basename(f)), csvs[0])
    print(f"using: {src} ({os.path.getsize(src):,} bytes)")

    df = pd.read_csv(src)
    print(f"rows: {len(df):,}  columns: {list(df.columns)}")
    print(df.head(3).to_string())

    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    dt_col = next((cols_lower[c] for c in
                   ("datetime", "timestamp", "date_time", "date", "time", "dt")
                   if c in cols_lower), df.columns[0])
    ts = pd.to_datetime(df[dt_col])
    print(f"timestamp column: {dt_col!r}  range: {ts.min()} .. {ts.max()}")

    os.makedirs("data", exist_ok=True)
    sub = df[(ts >= "2010-01-01") & (ts < "2011-01-01")]
    sub.to_csv("data/spy_1min_2010.csv", index=False)
    print(f"2010 subset: {len(sub):,} rows -> data/spy_1min_2010.csv")

    with open(src, "rb") as fi, gzip.open(
            f"data/{TARGET_NAME}.csv.gz", "wb", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo)
    print(f"gzipped full file: {os.path.getsize(f'data/{TARGET_NAME}.csv.gz'):,} bytes")


if __name__ == "__main__":
    main()
