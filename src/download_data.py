# src/download_data.py

import pandas as pd
import yfinance as yf
from pathlib import Path


# =====================================================
# CONFIG
# =====================================================

START_DATE = "1970-01-01"
END_DATE = "2024-01-01"

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# DOWNLOAD S&P 500
# =====================================================

def download_sp500():
    print("Downloading S&P 500 data...")

    sp500 = yf.download(
        "^GSPC",
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
    )

    if sp500.empty:
        raise RuntimeError("No S&P 500 data was downloaded.")

    sp500.to_csv(RAW_DATA_DIR / "sp500.csv")

    print(f"Saved {len(sp500)} observations")


# =====================================================
# DOWNLOAD FRED DATA
# =====================================================

def download_fred_series(fred_code):
    data = pd.read_csv(
        f"{FRED_CSV_URL}?id={fred_code}",
        parse_dates=["observation_date"],
        na_values=".",
    )

    data = data.rename(
        columns={
            "observation_date": "date",
            fred_code: "value",
        }
    )
    data = data.set_index("date").sort_index()

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    return data.loc[(data.index >= start) & (data.index < end), "value"]


def download_fred():
    print("Downloading FRED data...")

    fred_series = [
        "UNRATE",
        "FEDFUNDS",
        "CPIAUCSL",
        "INDPRO",
        "AAA",
        "BAA",
        "GS10",
        "TB3MS",
        "CP3M",
        "CPN3M",
        "GDPC1",
        "PPIACO",
    ]

    macro = pd.DataFrame()

    for fred_code in fred_series:

        print(f"Downloading {fred_code}")

        macro[fred_code] = download_fred_series(fred_code)

    if macro.empty:
        raise RuntimeError("No FRED data was downloaded.")

    macro.to_csv(RAW_DATA_DIR / "fred_macro.csv")

    print("Saved macroeconomic data")


# =====================================================
# MAIN
# =====================================================

def main():

    download_sp500()
    download_fred()

    print("\nFinished downloading all datasets.")


if __name__ == "__main__":
    main()
