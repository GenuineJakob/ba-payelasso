# src/download_data.py

from pathlib import Path

import pandas as pd

try:
    import tidyfinance as tf
except ImportError as exc:
    raise ImportError(
        "tidyfinance is required to download the data. Install it with "
        "`pip install tidyfinance`."
    ) from exc


# =====================================================
# CONFIG
# =====================================================

START_DATE = "1970-01-01"
END_DATE = "2024-01-01"

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
SP500_SYMBOL = "^GSPC"
FRED_SERIES = [
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

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# DOWNLOAD S&P 500
# =====================================================

def download_sp500():
    print("Downloading S&P 500 data...")

    sp500 = tf.download_data(
        domain="Stock Prices",
        symbols=[SP500_SYMBOL],
        start_date=START_DATE,
        end_date=END_DATE,
    )

    if sp500.empty:
        raise RuntimeError("No S&P 500 data was downloaded.")

    # Keep the on-disk contract used by build_dataset.py. tidyfinance returns
    # tidy, lower-case columns and both raw and adjusted closing prices. The
    # previous yfinance call used auto_adjust=True, so its Close column was the
    # adjusted close.
    sp500["date"] = pd.to_datetime(sp500["date"])
    sp500 = sp500.loc[
        (sp500["date"] >= pd.Timestamp(START_DATE))
        & (sp500["date"] < pd.Timestamp(END_DATE))
    ].copy()
    sp500 = sp500.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Unadjusted Close",
            "adjusted_close": "Close",
            "volume": "Volume",
        }
    )
    sp500 = sp500.drop(columns=["symbol"], errors="ignore")
    sp500 = sp500.set_index("date").sort_index()

    if "Close" not in sp500.columns or sp500["Close"].notna().sum() == 0:
        raise RuntimeError("Downloaded S&P 500 data contain no adjusted prices.")

    sp500.to_csv(RAW_DATA_DIR / "sp500.csv", index_label="Date")

    print(f"Saved {len(sp500)} observations")


# =====================================================
# DOWNLOAD FRED DATA
# =====================================================

def download_fred():
    print("Downloading FRED data...")

    macro_long = tf.download_data(
        domain="FRED",
        series=FRED_SERIES,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    if macro_long.empty:
        raise RuntimeError("No FRED data was downloaded.")

    required_columns = {"date", "series", "value"}
    missing_columns = required_columns.difference(macro_long.columns)
    if missing_columns:
        raise RuntimeError(
            "Unexpected tidyfinance FRED schema; missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    macro_long["date"] = pd.to_datetime(macro_long["date"])
    macro_long = macro_long.loc[
        (macro_long["date"] >= pd.Timestamp(START_DATE))
        & (macro_long["date"] < pd.Timestamp(END_DATE))
    ].copy()
    macro_long["value"] = pd.to_numeric(macro_long["value"], errors="coerce")

    # build_dataset.py expects one FRED series per column.
    macro = (
        macro_long.pivot(index="date", columns="series", values="value")
        .reindex(columns=FRED_SERIES)
        .sort_index()
    )
    macro.columns.name = None

    missing_series = [code for code in FRED_SERIES if macro[code].notna().sum() == 0]
    if missing_series:
        print("Warning: no observations downloaded for " + ", ".join(missing_series))

    macro.to_csv(RAW_DATA_DIR / "fred_macro.csv", index_label="date")

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
