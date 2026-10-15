"""Build separate same-month explanation and next-quarter forecast datasets.

Run from Code: python src/build_volatility_analysis_datasets.py
Existing datasets and forecasting results are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_fredmd_dataset import (
    apply_transformation, feature_pools, read_dictionary, read_snapshot,
    require_monthly_index, TRANSFORM_NAMES,
)
from download_fredmd import sha256_file
from fredmd_config import (
    ALPHAS, APPENDIX_NAME, BASELINE_FEATURES, BASE_DATA_PATH,
    PROCESSED_DIR, RAW_DIR, SNAPSHOT_NAME, SOURCE_MANIFEST_NAME, VINTAGE,
)

SETTINGS = {
    "contemporaneous": {
        "window_size": 240, "controls": [f"vol_lag_{k}" for k in range(1, 7)],
        "frequency": "ME", "first_origin": "2001-01-31",
        "periods": {"main": ("2001-01-31", "2020-12-31"),
                    "later": ("2021-01-31", "2023-11-30")},
        "interpretation": "Ex-post same-month explanation; neither real-time forecasting nor causal identification.",
    },
    "quarterly": {
        "window_size": 80, "controls": ["vol", "vol_lag_1"],
        "frequency": "QE-DEC", "first_origin": "2000-12-31",
        "periods": {"main": ("2000-12-31", "2020-09-30"),
                    "later": ("2020-12-31", "2023-09-30")},
        "interpretation": "Next-quarter forecasting with previous-quarter macro and current-quarter financial information.",
    },
}
CAVEAT = (
    "Revised-data experiments, not historical real-time reconstructions. "
    "The later period is a retrospective robustness sample. Daily S&P 500 log "
    "index returns are retained; lag orders and log targets follow Paye, not an exact replication."
)


def freeze_features(data: pd.DataFrame, metadata: pd.DataFrame, kind: str) -> pd.DataFrame:
    settings = SETTINGS[kind]
    first = pd.Timestamp(settings["first_origin"])
    train = data.loc[data.index < first].tail(settings["window_size"])
    if len(train) != settings["window_size"] or not np.isfinite(
        train[["target"] + settings["controls"]].to_numpy()
    ).all():
        raise ValueError(f"Incomplete initial {kind} training history.")
    result = metadata.copy()
    result["initial_training_start"] = train.index[0].strftime("%Y-%m-%d")
    result["initial_training_end"] = train.index[-1].strftime("%Y-%m-%d")
    result["initial_training_n"] = len(train)
    result["initial_finite_n"] = [int(np.isfinite(train[f]).sum()) for f in result.feature]
    result["initial_coverage"] = result.initial_finite_n / len(train)
    result["eligible"] = result.initial_coverage >= .95
    result["exclusion_reason"] = np.where(result.eligible, "", "initial_coverage_below_95_percent")
    return result


def contemporaneous_features(base, raw, codes, dictionary):
    """Use reference-month values, not the delayed monthly forecast dataset."""
    require_monthly_index(base, "Baseline monthly data")
    require_monthly_index(raw, "FRED-MD snapshot")
    data = pd.DataFrame({"target": base.volatility}, index=base.index)
    for k in range(1, 7):
        data[f"vol_lag_{k}"] = base.volatility.shift(k)
    rows, features = [], {}
    for name, (description, group, _, transformation) in BASELINE_FEATURES.items():
        feature = f"core__{name}"
        features[feature] = base[name].where(np.isfinite(base[name]))
        rows.append({"feature": feature, "source": "current_monthly", "source_code": name,
                     "description": description, "group": group, "tcode": None,
                     "transformation": transformation, "reporting_lag_months": 0})
    for name in raw:
        info = dictionary[name.casefold()]
        feature = f"fredmd__{name}"
        features[feature] = apply_transformation(raw[name], codes[name]).reindex(base.index)
        rows.append({"feature": feature, "source": "fredmd", "source_code": name,
                     **info, "tcode": codes[name], "transformation": TRANSFORM_NAMES[codes[name]],
                     "reporting_lag_months": 0})
    data = pd.concat([data, pd.DataFrame(features, index=data.index)], axis=1)
    data["target_date"] = data.index
    data.index.name = "date"
    meta = freeze_features(data, pd.DataFrame(rows), "contemporaneous")
    return data, meta, feature_pools(meta)


def read_prices(path: Path) -> pd.Series:
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    column = "Close" if "Close" in raw else "Adj Close"
    if column not in raw:
        raise ValueError("Daily prices need Close or Adj Close.")
    prices = pd.Series(pd.to_numeric(raw[column], errors="coerce").to_numpy(), index=dates)
    prices = prices.loc[prices.index.notna()].sort_index()
    if not prices.index.is_unique or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Daily prices must have unique dates and finite, positive values.")
    return prices


def quarterly_volatility(prices: pd.Series) -> pd.Series:
    """Compute returns once, then sum squared returns within complete quarters.

    The first quarter lacks its opening return and is excluded. Terminal
    coverage must reach the last business day (allowing a weekday holiday).
    All three calendar months must contain daily observations.
    """
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("Daily price dates must be unique and sorted.")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Daily prices must be finite and positive.")
    returns = np.log(prices).diff()
    squared = returns.pow(2).resample("QE-DEC").sum(min_count=1)
    counts = prices.resample("ME").count().resample("QE-DEC").apply(lambda s: int((s > 0).sum()))
    vol = np.sqrt(4 * squared).where(counts == 3)
    vol.iloc[0] = np.nan
    terminal = prices.index[-1].to_period("Q").end_time.normalize()
    last_business = terminal - pd.offsets.BDay(0)
    if prices.index[-1] < last_business - pd.offsets.BDay(1):
        vol.iloc[-1] = np.nan
    return vol.rename("volatility")


def complete_quarter_means(monthly: pd.DataFrame) -> pd.DataFrame:
    require_monthly_index(monthly, "Raw macro monthly data")
    finite = monthly.where(np.isfinite(monthly))
    return finite.resample("QE-DEC").mean().where(finite.resample("QE-DEC").count() == 3)


def quarterly_macro_features(raw: pd.DataFrame):
    """Quarterly transformations precede the explicit availability shift."""
    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index) + pd.offsets.MonthEnd(0)
    raw = raw.sort_index()
    monthly = pd.DataFrame(index=raw.index)
    monthly["unemployment"] = raw.UNRATE
    monthly["fed_funds"] = raw.FEDFUNDS
    monthly["default_spread"] = raw.BAA - raw.AAA
    monthly["term_spread"] = raw.GS10 - raw.TB3MS
    cp = raw.CPN3M.combine_first(raw.CP3M)
    monthly["commercial_paper_spread"] = cp - raw.TB3MS
    for source, name in [("CPIAUCSL", "cpi"), ("INDPRO", "ip"), ("PPIACO", "ppi")]:
        monthly[name] = raw[source]
    q = complete_quarter_means(monthly)
    log_indices = np.log(q[["cpi", "ip", "ppi"]].where(q[["cpi", "ip", "ppi"]] > 0))
    growth = log_indices.diff(4)
    q["inflation"] = growth.cpi
    q["ip_growth"] = growth.ip
    q["ip_volatility"] = growth.ip.rolling(4, min_periods=4).std(ddof=1)
    q["ppi_volatility"] = growth.ppi.rolling(4, min_periods=4).std(ddof=1)
    rows, features = [], {}
    for name, (description, group, _, _) in BASELINE_FEATURES.items():
        delay = 0 if group == 6 else 1
        feature = f"core__{name}"
        features[feature] = q[name].shift(delay)
        transformation = (
            "four-quarter sample SD of four-quarter log index changes" if name.endswith("volatility")
            else "four-quarter log change of quarterly mean index" if name in {"inflation", "ip_growth"}
            else "mean of three finite monthly observations"
        )
        rows.append({"feature": feature, "source": "current_quarterly", "source_code": name,
                     "description": description, "group": group,
                     "transformation": transformation, "reporting_lag_quarters": delay})
    return pd.DataFrame(features), pd.DataFrame(rows)


def quarterly_features(prices, macro):
    vol = quarterly_volatility(prices)
    features, meta = quarterly_macro_features(macro)
    data = pd.DataFrame({"vol": vol, "vol_lag_1": vol.shift(1), "target": vol.shift(-1)})
    data = data.join(features)
    data["target_date"] = data.index + pd.offsets.QuarterEnd(startingMonth=12)
    # Retain a consecutive calendar; only the structural boundary quarters go.
    data = data.iloc[1:-1].copy()
    data.index.name = "date"
    meta = freeze_features(data, meta, "quarterly")
    pools = {"current_quarterly": meta.loc[meta.eligible, "feature"].tolist()}
    return data, meta, pools


def build_datasets(raw_dir=RAW_DIR, base_path=BASE_DATA_PATH, processed_dir=PROCESSED_DIR):
    sources = json.loads((raw_dir / SOURCE_MANIFEST_NAME).read_text())
    if sources["vintage"] != VINTAGE:
        raise ValueError("Unexpected FRED-MD vintage.")
    for name in [SNAPSHOT_NAME, APPENDIX_NAME]:
        if sha256_file(raw_dir / name) != sources["artifacts"][name]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
    base = pd.read_csv(base_path, index_col=0, parse_dates=True)
    raw, codes = read_snapshot(raw_dir / SNAPSHOT_NAME)
    dictionary = read_dictionary(raw_dir / APPENDIX_NAME)
    prices = read_prices(raw_dir / "sp500.csv")
    macro = pd.read_csv(raw_dir / "fred_macro.csv", index_col=0, parse_dates=True)
    checksums = {str(p.resolve()): sha256_file(p) for p in [
        base_path, raw_dir / SNAPSHOT_NAME, raw_dir / APPENDIX_NAME,
        raw_dir / SOURCE_MANIFEST_NAME, raw_dir / "sp500.csv", raw_dir / "fred_macro.csv",
    ]}
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "contemporaneous": contemporaneous_features(base, raw, codes, dictionary),
        "quarterly": quarterly_features(prices, macro),
    }
    for kind, (data, metadata, pools) in outputs.items():
        dataset_path = processed_dir / f"{kind}_dataset.csv"
        metadata_path = processed_dir / f"{kind}_feature_metadata.csv"
        data.to_csv(dataset_path)
        metadata.to_csv(metadata_path, index=False)
        manifest = {
            "kind": kind, **SETTINGS[kind], "caveat": CAVEAT, "vintage": VINTAGE,
            "coverage_threshold": .95, "pools": pools, "sources": checksums,
            "dataset": {"file": dataset_path.name, "sha256": sha256_file(dataset_path)},
            "metadata": {"file": metadata_path.name, "sha256": sha256_file(metadata_path)},
            "alphas": ALPHAS.tolist(), "data_n": len(data),
        }
        (processed_dir / f"{kind}_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Built {kind}: {len(data)} rows; pools { {p: len(f) for p, f in pools.items()} }")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--base-data", type=Path, default=BASE_DATA_PATH)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    build_datasets(args.raw_dir, args.base_data, args.processed_dir)


if __name__ == "__main__":
    main()
