# src/build_dataset.py

import pandas as pd
import numpy as np
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

# Project root is the Code directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# FRED SERIES DEFINITIONS
# =====================================================

FRED_COLUMN_MAP = {
    "UNRATE": "unemployment",
    "FEDFUNDS": "fed_funds",
    "CPIAUCSL": "cpi",
    "INDPRO": "industrial_production",
    "AAA": "aaa_yield",
    "BAA": "baa_yield",
    "GS10": "ten_year",
    "TB3MS": "three_month",
    "CP3M": "commercial_paper_3m_historical",
    "CPN3M": "commercial_paper_3m_modern",
    "GDPC1": "real_gdp",
    "PPIACO": "ppi",
}


# =====================================================
# LOAD RAW DATA
# =====================================================

def load_sp500():
    sp500_path = RAW_DATA_DIR / "sp500.csv"
    if not sp500_path.exists():
        raise FileNotFoundError(
            f"Could not find {sp500_path}. Run `python src/download_data.py` first from inside the Code directory "
            "or check whether sp500.csv exists in data/raw/."
        )

    # yfinance sometimes writes extra header rows when saving downloaded data.
    # Therefore we load the CSV flexibly, coerce the first column to dates,
    # drop non-date rows, and convert all price columns to numeric values.
    sp500 = pd.read_csv(sp500_path)

    date_col = sp500.columns[0]
    sp500[date_col] = pd.to_datetime(sp500[date_col], errors="coerce")
    sp500 = sp500.dropna(subset=[date_col])
    sp500 = sp500.set_index(date_col)

    for col in sp500.columns:
        sp500[col] = pd.to_numeric(sp500[col], errors="coerce")

    sp500 = sp500.sort_index()

    return sp500


def load_fred():
    fred_path = RAW_DATA_DIR / "fred_macro.csv"
    if not fred_path.exists():
        raise FileNotFoundError(
            f"Could not find {fred_path}. Run `python src/download_data.py` first from inside the Code directory "
            "or check whether fred_macro.csv exists in data/raw/."
        )
    macro = pd.read_csv(
        fred_path,
        index_col=0,
        parse_dates=True
    )

    return macro


def load_goyal_welch():
    goyal_path = RAW_DATA_DIR / "goyal-welch-a.csv"
    if not goyal_path.exists():
        print(f"Optional file not found: {goyal_path}. Skipping Goyal-Welch variables.")
        return pd.DataFrame()

    data = pd.read_csv(goyal_path)
    data["date"] = pd.to_datetime(data["yyyy"].astype(str) + "-12-31")
    data = data.set_index("date").sort_index()

    for col in ["corprate", "ltrate", "cay", "ik"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    predictors = pd.DataFrame(index=data.index)
    predictors["dfr"] = data["corprate"] - data["ltrate"]
    predictors["cay"] = data["cay"]
    predictors["ik"] = data["ik"]

    return predictors


def load_payout_data():
    payout_path = RAW_DATA_DIR / "PayoutPaperTS_04_Mar_2021_monthly.csv"
    if not payout_path.exists():
        print(f"Optional file not found: {payout_path}. Skipping payout variables.")
        return pd.DataFrame()

    data = pd.read_csv(
        payout_path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )
    data["date"] = pd.to_datetime(
        data["year"].astype(str) + "-" + data["month"].astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)
    data = data.set_index("date").sort_index()

    data["net_pay_yld"] = pd.to_numeric(data["net_pay_yld"], errors="coerce")

    predictors = pd.DataFrame(index=data.index)
    predictors["npy"] = data["net_pay_yld"]

    return predictors


# =====================================================
# BUILD VOLATILITY
# =====================================================

def build_monthly_volatility(sp500):
    """
    Computes monthly realized volatility from daily S&P 500 prices.
    """

    if "Close" in sp500.columns:
        price_col = "Close"
    elif "Adj Close" in sp500.columns:
        price_col = "Adj Close"
    else:
        raise ValueError("No Close or Adj Close column found in S&P 500 data.")

    sp500[price_col] = pd.to_numeric(sp500[price_col], errors="coerce")
    sp500 = sp500.dropna(subset=[price_col])

    if sp500.empty:
        raise ValueError(
            f"No valid numeric observations found in column {price_col}. "
            "Check data/raw/sp500.csv."
        )

    sp500["log_return"] = np.log(sp500[price_col]).diff()

    monthly_rv = sp500["log_return"].pow(2).resample("ME").sum()

    monthly_vol = np.sqrt(12 * monthly_rv)
    monthly_vol.name = "volatility"

    return monthly_vol


# =====================================================
# BUILD MACRO VARIABLES
# =====================================================

def build_macro_variables(macro):
    """
    Constructs macroeconomic predictors from FRED data.
    """

    macro = macro.rename(columns=FRED_COLUMN_MAP)
    macro = macro.resample("ME").last().ffill()

    if "commercial_paper_3m" not in macro.columns:
        macro["commercial_paper_3m"] = np.nan

    if "commercial_paper_3m_historical" in macro.columns:
        macro["commercial_paper_3m"] = macro["commercial_paper_3m"].combine_first(
            macro["commercial_paper_3m_historical"]
        )

    if "commercial_paper_3m_modern" in macro.columns:
        macro["commercial_paper_3m"] = macro["commercial_paper_3m_modern"].combine_first(
            macro["commercial_paper_3m"]
        )

    macro["default_spread"] = macro["baa_yield"] - macro["aaa_yield"]
    macro["term_spread"] = macro["ten_year"] - macro["three_month"]
    macro["commercial_paper_spread"] = (
        macro["commercial_paper_3m"] - macro["three_month"]
    )

    macro["inflation"] = np.log(macro["cpi"]).diff(12)
    macro["ip_growth"] = np.log(macro["industrial_production"]).diff(12)
    macro["gdp_growth"] = np.log(macro["real_gdp"]).diff(12)
    macro["ppi_inflation"] = np.log(macro["ppi"]).diff(12)

    macro["ip_volatility"] = macro["ip_growth"].rolling(12).std()
    macro["ppi_volatility"] = macro["ppi_inflation"].rolling(12).std()

    predictors = [
        "unemployment",
        "fed_funds",
        "default_spread",
        "term_spread",
        "commercial_paper_spread",
        "inflation",
        "ip_growth",
        "gdp_growth",
        "ppi_volatility",
        "ip_volatility",
    ]

    return macro[predictors]


def build_external_variables(goyal_welch, payout):
    """
    Builds non-FRED predictors from Goyal-Welch and Roberts payout data.
    """

    external_data = []

    if not goyal_welch.empty:
        goyal_monthly = goyal_welch.resample("ME").last().ffill()
        external_data.append(goyal_monthly)

    if not payout.empty:
        payout_monthly = payout.resample("ME").last()
        external_data.append(payout_monthly)

    if not external_data:
        return pd.DataFrame()

    return pd.concat(external_data, axis=1, sort=False)


# =====================================================
# MERGE DATASET
# =====================================================

def build_final_dataset(monthly_vol, macro):
    """
    Builds final prediction dataset.

    Target:
        volatility_next_month

    Predictors:
        volatility at time t
        macro variables at time t
    """

    data = pd.concat([monthly_vol, macro], axis=1, sort=False)

    data["target"] = data["volatility"].shift(-1)

    required_columns = [
        "volatility",
        "target",
        "unemployment",
        "fed_funds",
        "default_spread",
        "term_spread",
        "inflation",
        "ip_growth",
    ]
    data = data.dropna(subset=required_columns)

    return data


# =====================================================
# MAIN
# =====================================================

def main():
    print("Loading raw data...")

    sp500 = load_sp500()
    macro_raw = load_fred()
    goyal_welch_raw = load_goyal_welch()
    payout_raw = load_payout_data()

    print("Building monthly realized volatility...")
    monthly_vol = build_monthly_volatility(sp500)

    print("Building macroeconomic variables...")
    macro = build_macro_variables(macro_raw)

    print("Building external predictor variables...")
    external = build_external_variables(goyal_welch_raw, payout_raw)

    if not external.empty:
        macro = pd.concat([macro, external], axis=1, sort=False)

    print("Merging final dataset...")
    dataset = build_final_dataset(monthly_vol, macro)

    output_path = PROCESSED_DATA_DIR / "dataset.csv"
    dataset.to_csv(output_path)

    print(f"Dataset saved to: {output_path}")
    print(f"Final dataset shape: {dataset.shape}")
    print(dataset.head())


if __name__ == "__main__":
    main()
