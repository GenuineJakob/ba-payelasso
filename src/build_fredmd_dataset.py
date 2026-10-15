"""Build a separate monthly FRED-MD experiment dataset and frozen feature pools.

Usage: python src/build_fredmd_dataset.py
The original dataset.csv and previous experiment outputs are never changed.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from download_fredmd import sha256_file
from fredmd_config import (
    APPENDIX_NAME, AR_CONTROLS, BASE_DATA_PATH, BASELINE_FEATURES,
    BUILD_MANIFEST_NAME, COVERAGE_THRESHOLD, DATA_CAVEAT, DATASET_NAME,
    MACRO_GROUPS, METADATA_NAME, PERIODS, PROCESSED_DIR, RAW_DIR, SNAPSHOT_NAME,
    SOURCE_MANIFEST_NAME, TARGET, VINTAGE, WINDOW_SIZE,
)


TRANSFORM_NAMES = {
    1: "level", 2: "first difference", 3: "second difference", 4: "log level",
    5: "first log difference", 6: "second log difference",
    7: "first difference of fractional growth",
}


def require_monthly_index(frame: pd.DataFrame, label: str) -> None:
    if frame.empty or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} must have unique, sorted monthly observations.")
    expected = pd.date_range(frame.index[0], frame.index[-1], freq="ME")
    if not frame.index.equals(expected):
        raise ValueError(f"{label} must use consecutive month ends; do not bridge missing calendar months.")


def read_snapshot(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = pd.read_csv(path)
    if raw.columns[0] != "sasdate" or str(raw.iloc[0, 0]).strip() != "Transform:":
        raise ValueError("Expected sasdate followed by a Transform: row, not a data observation.")
    if not raw.columns.is_unique:
        raise ValueError("FRED-MD columns must be unique.")
    codes = {}
    for column in raw.columns[1:]:
        value = float(raw.iloc[0][column])
        if not np.isfinite(value) or value != int(value) or int(value) not in TRANSFORM_NAMES:
            raise ValueError(f"Invalid transformation code for {column}: {value}")
        codes[column] = int(value)
    data = raw.iloc[1:].copy()
    data["sasdate"] = pd.to_datetime(data["sasdate"], errors="raise") + pd.offsets.MonthEnd(0)
    data = data.set_index("sasdate").apply(pd.to_numeric, errors="coerce").astype(float)
    data.index.name = "date"
    require_monthly_index(data, "FRED-MD snapshot")
    return data, codes


def read_dictionary(path: Path) -> dict[str, dict]:
    with ZipFile(path) as archive:
        contents = archive.read("FRED-MD Appendix/FRED-MD_updated_appendix.csv")
    appendix = pd.read_csv(BytesIO(contents), encoding="cp1252")
    required = {"fred", "description", "group"}
    if not required.issubset(appendix.columns):
        raise ValueError("Appendix lacks series descriptions or official groups.")
    dictionary = {}
    for _, row in appendix.iterrows():
        key = str(row["fred"]).casefold()
        if key in dictionary:
            raise ValueError(f"Ambiguous case-insensitive appendix mapping: {key}")
        dictionary[key] = {"description": str(row["description"]), "group": int(row["group"])}
    return dictionary


def apply_transformation(series: pd.Series, code: int) -> pd.Series:
    """Apply the official tcode, using only this observation and its past."""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.where(np.isfinite(values))
    if code == 1:
        result = values
    elif code == 2:
        result = values.diff()
    elif code == 3:
        result = values.diff().diff()
    elif code in {4, 5, 6}:
        result = np.log(values.where(values > 0))
        for _ in range(code - 4):
            result = result.diff()
    elif code == 7:
        denominator = values.shift(1).where(values.shift(1) != 0)
        result = (values / denominator - 1.0).diff()
    else:
        raise ValueError(f"Unsupported transformation code: {code}")
    return result.where(np.isfinite(result))


def reporting_lag(source_code: str, group: int) -> int:
    key = source_code.casefold()
    if key in {"s&p div yield", "s&p pe ratio"}:
        return 6
    if group == 6 or key in {"s&p 500", "vixclsx"}:
        return 0
    if group in MACRO_GROUPS:
        return 2
    raise ValueError(f"No declared reporting delay for {source_code} in group {group}.")


def build_features(
    base: pd.DataFrame, raw: pd.DataFrame, codes: dict[str, int], dictionary: dict[str, dict]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_monthly_index(base, "Existing target dataset")
    require_monthly_index(raw, "FRED-MD snapshot")
    needed = {"volatility", TARGET, *BASELINE_FEATURES}
    if not needed.issubset(base.columns):
        raise ValueError(f"Missing baseline columns: {sorted(needed.difference(base.columns))}")
    if not np.isfinite(base[["volatility", TARGET]].to_numpy()).all():
        raise ValueError("Existing volatility observations and targets must be finite.")
    dataset = base[["volatility", TARGET]].copy()
    dataset["vol"] = base["volatility"]
    for lag in range(1, 6):
        dataset[f"vol_lag_{lag}"] = base["volatility"].shift(lag)
    rows = []
    features = {}
    for name, (description, group, lag, transformation) in BASELINE_FEATURES.items():
        feature = f"core__{name}"
        values = base[name].where(np.isfinite(base[name])).shift(lag)
        features[feature] = values
        rows.append({
            "feature": feature, "source": "current_monthly", "source_code": name,
            "description": description, "group": group, "tcode": None,
            "transformation": transformation, "reporting_lag_months": lag,
            "raw_missing": int(base[name].isna().sum()), "invalid_log_values": 0,
            "zero_growth_denominators": 0, "transformed_missing": int(values.isna().sum()),
        })
    for name in raw:
        info = dictionary.get(name.casefold())
        if info is None:
            raise ValueError(f"No official variable description/group for {name}.")
        code = codes[name]
        lag = reporting_lag(name, info["group"])
        transformed = apply_transformation(raw[name], code)
        feature = f"fredmd__{name}"
        features[feature] = transformed.shift(lag).reindex(base.index)
        rows.append({
            "feature": feature, "source": "fredmd", "source_code": name,
            "description": info["description"], "group": info["group"], "tcode": code,
            "transformation": TRANSFORM_NAMES[code], "reporting_lag_months": lag,
            "raw_missing": int(raw[name].isna().sum()),
            "invalid_log_values": int((raw[name] <= 0).sum()) if code in {4, 5, 6} else 0,
            "zero_growth_denominators": int((raw[name].shift(1) == 0).sum()) if code == 7 else 0,
            "transformed_missing": int(transformed.isna().sum()),
        })
    dataset = pd.concat([dataset, pd.DataFrame(features, index=base.index)], axis=1)
    dataset.index.name = "date"
    return dataset, pd.DataFrame(rows)


def freeze_eligibility(
    dataset: pd.DataFrame, metadata: pd.DataFrame,
    first_forecast: str = PERIODS["main"][0], window_size: int = WINDOW_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_forecast = pd.Timestamp(first_forecast)
    if first_forecast not in dataset.index:
        raise ValueError(f"Initial forecast origin {first_forecast.date()} is absent.")
    training = dataset.loc[dataset.index < first_forecast].tail(window_size)
    if len(training) != window_size or not np.isfinite(training[[TARGET] + AR_CONTROLS]).all().all():
        raise ValueError("Initial feature selection needs 240 consecutive, complete AR/target training rows.")
    result = metadata.copy()
    result["initial_training_start"] = training.index[0].strftime("%Y-%m-%d")
    result["initial_training_end"] = training.index[-1].strftime("%Y-%m-%d")
    result["initial_training_n"] = window_size
    result["initial_finite_n"] = [
        int(np.isfinite(training[feature].to_numpy()).sum()) for feature in result.feature
    ]
    result["initial_coverage"] = result["initial_finite_n"] / window_size
    result["eligible"] = result["initial_coverage"] >= COVERAGE_THRESHOLD
    result["exclusion_reason"] = np.where(result.eligible, "", "initial_coverage_below_95_percent")
    return result, training


def feature_pools(metadata: pd.DataFrame) -> dict[str, list[str]]:
    retained = metadata.loc[metadata.eligible]
    fred = retained.loc[retained.source == "fredmd"]
    pools = {
        "current_monthly": retained.loc[retained.source == "current_monthly", "feature"].tolist(),
        "fredmd_macro": fred.loc[fred.group.isin(MACRO_GROUPS), "feature"].tolist(),
        "fredmd_no_iv": fred.loc[fred.source_code.str.casefold() != "vixclsx", "feature"].tolist(),
        "fredmd_full": fred.feature.tolist(),
    }
    implied = fred.loc[fred.source_code.str.casefold() == "vixclsx", "feature"].tolist()
    if len(implied) != 1 or set(pools["fredmd_full"]) - set(pools["fredmd_no_iv"]) != set(implied):
        raise ValueError("The eligible financial pools must differ by exactly VIXCLSx.")
    if any(not pool for pool in pools.values()):
        raise ValueError("Every declared predictor pool must contain eligible features.")
    return pools


def build_dataset(
    raw_dir: Path = RAW_DIR, base_path: Path = BASE_DATA_PATH,
    processed_dir: Path = PROCESSED_DIR,
) -> dict:
    source_path = raw_dir / SOURCE_MANIFEST_NAME
    if not source_path.exists():
        raise FileNotFoundError("Run python src/download_fredmd.py before building the dataset.")
    sources = json.loads(source_path.read_text())
    if sources["vintage"] != VINTAGE:
        raise ValueError("The source manifest is not the pinned September 2026 vintage.")
    for name in [SNAPSHOT_NAME, APPENDIX_NAME]:
        if sha256_file(raw_dir / name) != sources["artifacts"][name]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
    raw, codes = read_snapshot(raw_dir / SNAPSHOT_NAME)
    dictionary = read_dictionary(raw_dir / APPENDIX_NAME)
    base = pd.read_csv(base_path, index_col=0, parse_dates=True)
    dataset, metadata = build_features(base, raw, codes, dictionary)
    metadata, training = freeze_eligibility(dataset, metadata)
    pools = feature_pools(metadata)
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_path, metadata_path = processed_dir / DATASET_NAME, processed_dir / METADATA_NAME
    dataset.to_csv(dataset_path)
    metadata.to_csv(metadata_path, index=False)
    manifest = {
        "vintage": VINTAGE, "caveat": DATA_CAVEAT, "sources": sources,
        "baseline_source": {"file": base_path.name, "sha256": sha256_file(base_path)},
        "dataset": {"file": DATASET_NAME, "sha256": sha256_file(dataset_path)},
        "metadata": {"file": METADATA_NAME, "sha256": sha256_file(metadata_path)},
        "window_size": WINDOW_SIZE, "coverage_threshold": COVERAGE_THRESHOLD,
        "eligibility_training_start": training.index[0].strftime("%Y-%m-%d"),
        "eligibility_training_end": training.index[-1].strftime("%Y-%m-%d"),
        "periods": PERIODS, "pools": pools,
        "excluded_features": metadata.loc[~metadata.eligible, ["feature", "exclusion_reason"]].to_dict("records"),
        "dataset_n": len(dataset), "snapshot_predictors": len(raw.columns),
    }
    (processed_dir / BUILD_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved separate dataset: {dataset_path} ({len(dataset)} monthly rows)")
    print("Eligible pool sizes:", {name: len(pool) for name, pool in pools.items()})
    print("Excluded:", manifest["excluded_features"])
    print(DATA_CAVEAT)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--base-data", type=Path, default=BASE_DATA_PATH)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    build_dataset(args.raw_dir, args.base_data, args.processed_dir)


if __name__ == "__main__":
    main()
