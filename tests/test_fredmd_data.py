from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from build_fredmd_dataset import (
    apply_transformation, build_dataset, build_features, feature_pools,
    freeze_eligibility, read_dictionary, read_snapshot, reporting_lag,
)
import download_fredmd as download
from fredmd_config import (
    APPENDIX_NAME, AR_CONTROLS, BASELINE_FEATURES, DATASET_NAME,
    METADATA_NAME, SNAPSHOT_NAME, SOURCE_MANIFEST_NAME, VINTAGE,
)


@pytest.mark.parametrize("code,expected", [
    (1, [1, 2, 4, 8, 16]),
    (2, [np.nan, 1, 2, 4, 8]),
    (3, [np.nan, np.nan, 1, 2, 4]),
    (4, np.log([1, 2, 4, 8, 16])),
    (5, [np.nan, np.log(2), np.log(2), np.log(2), np.log(2)]),
    (6, [np.nan, np.nan, 0, 0, 0]),
    (7, [np.nan, np.nan, 0, 0, 0]),
])
def test_all_official_transformations(code, expected):
    actual = apply_transformation(pd.Series([1, 2, 4, 8, 16]), code)
    np.testing.assert_allclose(actual, expected, atol=1e-12, equal_nan=True)


def test_invalid_logs_divisions_and_negative_reserves_are_handled():
    result = apply_transformation(pd.Series([1, 0, -1, 2, np.inf, np.nan]), 4)
    np.testing.assert_allclose(result, [0, np.nan, np.nan, np.log(2), np.nan, np.nan], equal_nan=True)
    assert apply_transformation(pd.Series([1, 0, 2, 4]), 7).isna().all()
    np.testing.assert_allclose(
        apply_transformation(pd.Series([10, -5, -10, -20]), 7),
        [np.nan, np.nan, 2.5, 0], equal_nan=True,
    )


@pytest.fixture
def synthetic_sources():
    dates = pd.date_range("1980-01-31", "2023-11-30", freq="ME")
    rng = np.random.default_rng(51)
    volatility = np.maximum(0.02, 0.14 + rng.normal(scale=0.03, size=len(dates) + 1))
    base = pd.DataFrame({"volatility": volatility[:-1], "target": volatility[1:]}, index=dates)
    for column in BASELINE_FEATURES:
        base[column] = 3 + rng.normal(size=len(dates))
    specs = {
        "RPI": (5, 1), "UNRATE": (2, 2), "HOUST": (4, 3),
        "AMDMNOx": (5, 4), "M1SL": (6, 5), "FEDFUNDS": (2, 6),
        "CPIAUCSL": (6, 7), "S&P 500": (5, 8), "VIXCLSx": (1, 8),
        "S&P div yield": (2, 8), "S&P PE ratio": (5, 8),
        "ACOGNO": (5, 4), "IPB51222S": (5, 1),
    }
    raw = pd.DataFrame({
        column: np.exp(np.cumsum(rng.normal(scale=0.02, size=len(dates)))) * 10
        for column in specs
    }, index=dates)
    raw.loc[raw.index < "1992-02-29", "ACOGNO"] = np.nan
    # The March 2001 origin will have a missing delayed RPI predictor.
    raw.loc["2001-01-31", "RPI"] = np.nan
    codes = {name: code for name, (code, _) in specs.items()}
    dictionary = {name.casefold(): {"group": group, "description": name + " description"}
                  for name, (_, group) in specs.items()}
    return base, raw, codes, dictionary


@pytest.fixture
def frozen_experiment(tmp_path, synthetic_sources):
    base, raw, codes, dictionary = synthetic_sources
    raw_dir, processed = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir()
    processed.mkdir()
    base_path = processed / "dataset.csv"
    base.to_csv(base_path)
    transformed_row = pd.DataFrame([codes], index=["Transform:"])
    observations = raw.copy()
    observations.index = observations.index.strftime("%m/%d/%Y")
    snapshot = pd.concat([transformed_row, observations])
    snapshot.index.name = "sasdate"
    snapshot.to_csv(raw_dir / SNAPSHOT_NAME)
    descriptions = pd.DataFrame([
        {"fred": name.lower(), "description": dictionary[name.casefold()]["description"],
         "group": dictionary[name.casefold()]["group"], "tcode": 1}
        for name in raw
    ])
    with ZipFile(raw_dir / APPENDIX_NAME, "w") as archive:
        archive.writestr("FRED-MD Appendix/FRED-MD_updated_appendix.csv",
                         descriptions.to_csv(index=False).encode("cp1252"))
    sources = {"vintage": VINTAGE, "artifacts": {
        name: {"sha256": download.sha256_file(raw_dir / name)} for name in [SNAPSHOT_NAME, APPENDIX_NAME]
    }}
    (raw_dir / SOURCE_MANIFEST_NAME).write_text(json.dumps(sources))
    manifest = build_dataset(raw_dir, base_path, processed)
    return processed, manifest, base_path


def test_transform_row_case_insensitive_metadata_and_month_end_alignment(frozen_experiment):
    processed, manifest, base_path = frozen_experiment
    metadata = pd.read_csv(processed / METADATA_NAME).set_index("source_code")
    assert metadata.loc["IPB51222S", "description"] == "IPB51222S description"
    # Deliberately wrong appendix tcodes must not replace the snapshot's codes.
    assert metadata.loc["RPI", "tcode"] == 5
    data = pd.read_csv(processed / DATASET_NAME, index_col=0, parse_dates=True)
    base = pd.read_csv(base_path, index_col=0, parse_dates=True)
    pd.testing.assert_index_equal(data.index, base.index, check_names=False)
    assert data.index[0] == pd.Timestamp("1980-01-31")
    assert len(data) == len(base)
    pd.testing.assert_series_equal(data.target, base.target, check_names=False)
    assert not any(name in manifest["pools"]["current_monthly"] for name in ["dfr", "cay", "ik", "gdp_growth", "npy"])


def test_reporting_lags_and_initial_history_exclusions(synthetic_sources):
    base, raw, codes, dictionary = synthetic_sources
    data, metadata = build_features(base, raw, codes, dictionary)
    metadata, training = freeze_eligibility(data, metadata)
    assert training.index[0] == pd.Timestamp("1981-01-31")
    assert training.index[-1] == pd.Timestamp("2000-12-31")
    assert not metadata.set_index("source_code").loc["ACOGNO", "eligible"]
    for name in raw:
        lag = reporting_lag(name, dictionary[name.casefold()]["group"])
        expected = apply_transformation(raw[name], codes[name]).shift(lag)
        pd.testing.assert_series_equal(data[f"fredmd__{name}"], expected, check_names=False)
    for name, (_, _, lag, _) in BASELINE_FEATURES.items():
        pd.testing.assert_series_equal(data[f"core__{name}"], base[name].shift(lag), check_names=False)
    pools = feature_pools(metadata)
    assert set(pools["fredmd_full"]) - set(pools["fredmd_no_iv"]) == {"fredmd__VIXCLSx"}
    assert not {"fredmd__FEDFUNDS", "fredmd__S&P 500", "fredmd__VIXCLSx"}.intersection(pools["fredmd_macro"])
    assert data.loc["2001-03-31", "fredmd__RPI"] != data.loc["2001-03-31", "fredmd__RPI"]
    assert len(data) == len(base)


def test_future_changes_cannot_change_earlier_features_or_eligibility(synthetic_sources):
    base, raw, codes, dictionary = synthetic_sources
    original, metadata = build_features(base, raw, codes, dictionary)
    eligibility, _ = freeze_eligibility(original, metadata)
    changed_base, changed_raw = base.copy(), raw.copy()
    changed_base.loc[changed_base.index > "2001-03-31", list(BASELINE_FEATURES)] = np.nan
    changed_raw.loc[changed_raw.index > "2001-03-31"] = np.nan
    changed, changed_metadata = build_features(changed_base, changed_raw, codes, dictionary)
    changed_eligibility, _ = freeze_eligibility(changed, changed_metadata)
    pd.testing.assert_frame_equal(original.loc[:"2001-03-31"], changed.loc[:"2001-03-31"])
    columns = ["feature", "eligible", "initial_coverage", "exclusion_reason"]
    pd.testing.assert_frame_equal(eligibility[columns], changed_eligibility[columns])
    assert feature_pools(eligibility) == feature_pools(changed_eligibility)


def test_source_checksum_mismatch_is_rejected(frozen_experiment):
    processed, _, base_path = frozen_experiment
    raw_dir = processed.parent / "raw"
    with (raw_dir / SNAPSHOT_NAME).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Source checksum mismatch"):
        build_dataset(raw_dir, base_path, processed)


def test_snapshot_calendar_gaps_are_rejected(tmp_path):
    path = tmp_path / "snapshot.csv"
    pd.DataFrame({"sasdate": ["Transform:", "1/1/1980", "3/1/1980"], "RPI": [5, 1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="consecutive month ends"):
        read_snapshot(path)


def test_download_cache_and_tamper_detection(tmp_path, monkeypatch):
    snapshot = pd.DataFrame({"sasdate": ["Transform:", "1/1/1959"],
                             **{f"feature{i}": [1, i + 1] for i in range(126)}}).to_csv(index=False).encode()
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("FRED-MD Appendix/FRED-MD_updated_appendix.csv", "fred,description,group\nRPI,Income,1\n")
    contents = {download.SNAPSHOT_URL: snapshot, download.APPENDIX_URL: buffer.getvalue()}
    monkeypatch.setattr(download, "fetch_source", lambda url: contents[url])
    first = download.download_fredmd(tmp_path)
    monkeypatch.setattr(download, "fetch_source", lambda url: pytest.fail("Cached files must not be downloaded again"))
    assert download.download_fredmd(tmp_path) == first
    (tmp_path / SNAPSHOT_NAME).write_bytes(snapshot + b"\n")
    with pytest.raises(ValueError, match="integrity check"):
        download.download_fredmd(tmp_path)
