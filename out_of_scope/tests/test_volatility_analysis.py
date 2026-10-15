"""Timing, aggregation and independent metric checks for the new experiments."""

import json
import numpy as np
import pandas as pd
import pytest

from build_volatility_analysis_datasets import (
    SETTINGS, complete_quarter_means, contemporaneous_features, freeze_features,
    quarterly_features, quarterly_macro_features, quarterly_volatility,
)
from fredmd_config import BASELINE_FEATURES
from partially_penalized_lasso import PartiallyPenalizedLassoCV
from volatility_analysis_utils import (
    fit_ols, load_analysis, require_calendar, run_analysis, scale_volatility,
)
from download_fredmd import sha256_file
from sklearn.model_selection import TimeSeriesSplit


@pytest.fixture
def sources():
    rng = np.random.default_rng(218)
    dates = pd.date_range("1970-01-31", "2023-12-31", freq="ME")
    base = pd.DataFrame({"volatility": np.exp(rng.normal(-2, .35, len(dates)))}, index=dates)
    for name in BASELINE_FEATURES:
        base[name] = rng.normal(size=len(dates))
    specs = {"INDPRO": (5, 1), "UNRATE": (2, 2), "CPIAUCSL": (6, 7),
             "FEDFUNDS": (2, 6), "S&P 500": (5, 8), "VIXCLSx": (1, 8),
             "ACOGNO": (5, 4)}
    raw = pd.DataFrame({name: np.exp(rng.normal(size=len(dates)).cumsum() / 20)
                        for name in specs}, index=dates)
    raw.loc[raw.index < "1992-01-31", "ACOGNO"] = np.nan
    codes = {name: code for name, (code, _) in specs.items()}
    dictionary = {name.casefold(): {"description": name, "group": group}
                  for name, (_, group) in specs.items()}
    macro = pd.DataFrame({name: np.exp(rng.normal(scale=.03, size=len(dates)).cumsum()) * 10
                          for name in ["UNRATE", "FEDFUNDS", "AAA", "BAA", "GS10", "TB3MS",
                                       "CP3M", "CPN3M", "CPIAUCSL", "INDPRO", "PPIACO"]}, index=dates)
    days = pd.bdate_range("1970-01-01", "2023-12-29")
    prices = pd.Series(np.exp(rng.normal(scale=.01, size=len(days)).cumsum()) * 100, index=days)
    return base, raw, codes, dictionary, macro, prices


def test_same_month_features_and_target_have_no_current_vol_control(sources):
    base, raw, codes, dictionary, _, _ = sources
    data, meta, pools = contemporaneous_features(base, raw, codes, dictionary)
    date = pd.Timestamp("2001-01-31")
    assert data.loc[date, "target"] == base.loc[date, "volatility"]
    for lag in range(1, 7):
        assert data.loc[date, f"vol_lag_{lag}"] == base.volatility.shift(lag).loc[date]
    assert data.loc[date, "core__inflation"] == base.loc[date, "inflation"]
    assert data.loc[date, "fredmd__VIXCLSx"] == raw.loc[date, "VIXCLSx"]
    assert "vol" not in SETTINGS["contemporaneous"]["controls"]
    altered = base.copy()
    altered.loc[date, "volatility"] *= 10
    changed, _, _ = contemporaneous_features(altered, raw, codes, dictionary)
    np.testing.assert_array_equal(changed.loc[date, SETTINGS["contemporaneous"]["controls"]],
                                  data.loc[date, SETTINGS["contemporaneous"]["controls"]])
    excluded = meta.set_index("source_code").loc["ACOGNO"]
    assert not excluded.eligible
    assert set(pools["fredmd_full"]) - set(pools["fredmd_no_iv"]) == {"fredmd__VIXCLSx"}


def test_future_changes_do_not_change_earlier_features_or_eligibility(sources):
    base, raw, codes, dictionary, macro, prices = sources
    data, metadata, pools = contemporaneous_features(base, raw, codes, dictionary)
    modified = raw.copy()
    modified.loc[modified.index >= "2001-01-31"] = 500
    modified.loc[modified.index >= "2001-01-31", "ACOGNO"] = np.nan
    changed, changed_meta, changed_pools = contemporaneous_features(base, modified, codes, dictionary)
    pd.testing.assert_frame_equal(data.loc[:"2000-12-31"], changed.loc[:"2000-12-31"])
    pd.testing.assert_frame_equal(metadata, changed_meta)
    assert pools == changed_pools
    q, q_meta, q_pools = quarterly_features(prices, macro)
    future_macro = macro.copy()
    future_macro.loc[future_macro.index >= "2001-01-31"] *= 3
    future_prices = prices.copy()
    future_prices.loc[future_prices.index >= "2001-01-01"] *= 4
    changed_q, changed_q_meta, changed_q_pools = quarterly_features(future_prices, future_macro)
    # The 2000Q4 origin's target is 2001Q1; compare only earlier responses.
    pd.testing.assert_frame_equal(q.loc[:"2000-09-30"], changed_q.loc[:"2000-09-30"])
    pd.testing.assert_frame_equal(q_meta, changed_q_meta)
    assert q_pools == changed_q_pools


def test_quarterly_vol_matches_daily_returns_and_monthly_identity(sources):
    prices = sources[-1]
    quarterly = quarterly_volatility(prices)
    squared = np.log(prices).diff().pow(2)
    monthly_vol = np.sqrt(12 * squared.resample("ME").sum())
    rhs = monthly_vol.pow(2).resample("QE-DEC").sum() / 3
    assert np.isnan(quarterly.iloc[0])
    np.testing.assert_allclose(quarterly.iloc[1:].pow(2), rhs.iloc[1:])
    q2 = squared.loc["2001-04-01":"2001-06-30"].sum()
    assert quarterly.loc["2001-06-30"] == pytest.approx(np.sqrt(4 * q2))
    # Missing a whole month makes a quarter unavailable, without deleting it.
    missing = prices.loc[~((prices.index.year == 2001) & (prices.index.month == 5))]
    result = quarterly_volatility(missing)
    assert pd.isna(result.loc["2001-06-30"])
    pd.testing.assert_index_equal(result.index, quarterly.index)
    truncated = quarterly_volatility(prices.loc[:"2023-11-30"])
    assert pd.isna(truncated.iloc[-1])


def test_quarterly_transformations_availability_and_isolated_gaps(sources):
    macro = sources[-2]
    features, metadata = quarterly_macro_features(macro)
    q = macro.resample("QE-DEC").mean()
    ip_growth = np.log(q.INDPRO).diff(4)
    np.testing.assert_allclose(features["core__ip_growth"], ip_growth.shift(1), equal_nan=True)
    np.testing.assert_allclose(features["core__ip_volatility"], ip_growth.rolling(4).std().shift(1), equal_nan=True)
    np.testing.assert_allclose(features["core__fed_funds"], q.FEDFUNDS, equal_nan=True)
    assert metadata.set_index("source_code").loc["inflation", "reporting_lag_quarters"] == 1
    missing = macro.copy()
    missing.loc["2001-02-28", "UNRATE"] = np.nan
    changed, _ = quarterly_macro_features(missing)
    assert pd.isna(changed.loc["2001-06-30", "core__unemployment"])
    assert np.isfinite(changed.loc["2001-03-31", "core__unemployment"])
    assert len(changed) == len(features)
    # Prefer modern CP, fall back to historical CP for isolated missing months.
    missing.loc["2001-02-28", "CPN3M"] = np.nan
    changed, _ = quarterly_macro_features(missing)
    expected = (missing.CPN3M.combine_first(missing.CP3M) - missing.TB3MS).loc["2001-01-31":"2001-03-31"].mean()
    assert changed.loc["2001-03-31", "core__commercial_paper_spread"] == pytest.approx(expected)


def test_log_scale_transforms_target_and_controls_but_not_predictors(sources):
    data, _, _ = quarterly_features(sources[-1], sources[-2])
    controls = SETTINGS["quarterly"]["controls"]
    changed = scale_volatility(data, controls, "log")
    for column in ["target"] + controls:
        np.testing.assert_allclose(changed[column], np.log(data[column]), equal_nan=True)
    pd.testing.assert_series_equal(changed["core__inflation"], data["core__inflation"])
    data.loc[data.index[10], "target"] = 0
    with pytest.raises(ValueError, match="positive"):
        scale_volatility(data, controls, "log")


@pytest.mark.parametrize("kind", ["contemporaneous", "quarterly"])
def test_short_runner_dates_coefficients_and_independent_metrics(kind, sources, tmp_path):
    base, raw, codes, dictionary, macro, prices = sources
    if kind == "contemporaneous":
        data, metadata, pools = contemporaneous_features(base, raw, codes, dictionary)
    else:
        data, metadata, pools = quarterly_features(prices, macro)
    manifest = {"kind": kind, **SETTINGS[kind], "pools": pools, "caveat": "synthetic"}
    predictions, metrics = run_analysis(data, metadata, manifest, "log", "main",
                                        tmp_path / "results", tmp_path / "tables", max_forecasts=2)
    for (_, pool, model), group in predictions.groupby(["scale", "pool", "model"]):
        assert len(group) == 2
        mse = np.mean((group.actual.to_numpy() - group.prediction.to_numpy()) ** 2)
        benchmark_mse = np.mean((group.actual.to_numpy() - group.benchmark.to_numpy()) ** 2)
        row = metrics.loc[(metrics.pool == pool) & (metrics.model == model)].iloc[0]
        assert row.MSE == pytest.approx(mse)
        assert row.RMSE == pytest.approx(np.sqrt(mse))
        assert row.error_reduction_vs_AR == pytest.approx(1 - mse / benchmark_mse)
    coefficients = pd.read_csv(tmp_path / "results" / f"{kind}_coefficients_log_main_smoke.csv")
    controls = SETTINGS[kind]["controls"]
    assert not coefficients.loc[coefficients.feature.isin(controls + ["intercept"]), "penalized"].any()
    transformed = scale_volatility(data, controls, "log")
    for _, prediction in predictions.iterrows():
        coefs = coefficients.loc[(coefficients.date == prediction.date.strftime("%Y-%m-%d")) &
                                 (coefficients.pool == prediction.pool) & (coefficients.model == prediction.model)].set_index("feature").coefficient
        position = transformed.index.get_loc(prediction.date)
        train = transformed.iloc[position-SETTINGS[kind]["window_size"]:position]
        columns = coefs.index.drop("intercept")
        medians = train[columns.difference(controls)].median().fillna(0)
        X = transformed.loc[prediction.date, columns].fillna(medians)
        reconstructed = coefs["intercept"] + np.dot(coefs.loc[columns], X)
        assert prediction.prediction == pytest.approx(reconstructed)
    for _, group in predictions.groupby(["pool", "model"]):
        assert group.date.tolist() == predictions.date.drop_duplicates().tolist()
    if kind == "quarterly":
        assert predictions.target_date.min() == pd.Timestamp("2001-03-31")
        assert predictions.date.min() == pd.Timestamp("2000-12-31")
    else:
        assert predictions.date.equals(predictions.target_date)


def test_imputation_uses_training_only_and_retains_all_missing_column():
    dates = pd.date_range("1990-01-31", periods=80, freq="ME")
    rng = np.random.default_rng(24)
    train = pd.DataFrame({"target": rng.normal(size=80), "vol": rng.normal(size=80),
                          "x": rng.normal(size=80), "empty": np.nan}, index=dates)
    train.iloc[0, train.columns.get_loc("x")] = np.nan
    test = train.iloc[[-1]].copy()
    test["x"] = np.nan
    fitted = fit_ols(train, test, ["vol"], ["x", "empty"])
    changed = test.copy()
    changed["x"] = 1e9
    other = fit_ols(train, changed, ["vol"], ["x", "empty"])
    pd.testing.assert_series_equal(fitted["coefficients"], other["coefficients"])
    assert fitted["coefficients"]["empty"] == 0
    lasso = PartiallyPenalizedLassoCV(["vol"], np.array([.1, .01]), impute_missing=True, solver="lars")
    lasso.fit(train[["vol", "x", "empty"]], train.target)
    assert lasso.imputation_values_["x"] == train.x.median()
    assert lasso.coef_[2] == 0
    assert np.isfinite(lasso.predict(test[["vol", "x", "empty"]])).all()


def test_calendar_rejects_deleted_month(sources):
    base, raw, codes, dictionary, _, _ = sources
    data, _, _ = contemporaneous_features(base, raw, codes, dictionary)
    with pytest.raises(ValueError, match="calendar"):
        require_calendar(data.drop(data.index[10]), "contemporaneous")


def test_manifest_loader_checks_pools_and_generated_checksums(sources, tmp_path):
    data, metadata, pools = contemporaneous_features(*sources[:4])
    data_path = tmp_path / "contemporaneous_dataset.csv"
    meta_path = tmp_path / "contemporaneous_feature_metadata.csv"
    data.to_csv(data_path)
    metadata.to_csv(meta_path, index=False)
    manifest = {"kind": "contemporaneous", **SETTINGS["contemporaneous"],
                "pools": pools, "sources": {},
                "dataset": {"file": data_path.name, "sha256": sha256_file(data_path)},
                "metadata": {"file": meta_path.name, "sha256": sha256_file(meta_path)}}
    path = tmp_path / "contemporaneous_build_manifest.json"
    path.write_text(json.dumps(manifest))
    loaded, _, _ = load_analysis("contemporaneous", tmp_path)
    assert loaded.target_date.equals(pd.Series(loaded.index, index=loaded.index, name="target_date"))
    manifest["pools"]["fredmd_full"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="pools"):
        load_analysis("contemporaneous", tmp_path)
    data_path.write_text(data_path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_analysis("contemporaneous", tmp_path)


@pytest.mark.parametrize("n_controls", [2, 6])
def test_actual_solver_fold_preprocessing_is_unaffected_by_future_data(n_controls):
    rng = np.random.default_rng(19)
    X = pd.DataFrame(rng.normal(size=(100, n_controls + 2)),
                     columns=[f"ar{k}" for k in range(n_controls)] + ["x", "missing"])
    X.loc[::7, "missing"] = np.nan
    y = rng.normal(size=100)
    controls = X.columns[:n_controls].tolist()

    class RecordingLasso(PartiallyPenalizedLassoCV):
        def _fit_path(self, X, y, alphas):
            result = super()._fit_path(X, y, alphas)
            self.paths.append(result)
            return result

    train, _ = next(TimeSeriesSplit(n_splits=5).split(X))
    original = RecordingLasso(controls, np.array([.1, .01]), impute_missing=True, solver="lars")
    original.paths = []
    original.fit(X, y)
    future_X, future_y = X.copy(), y.copy()
    future_X.iloc[len(train):] += 10
    future_y[len(train):] += 5
    changed = RecordingLasso(controls, np.array([.1, .01]), impute_missing=True, solver="lars")
    changed.paths = []
    changed.fit(future_X, future_y)
    left, right = original.paths[0], changed.paths[0]
    pd.testing.assert_series_equal(left.imputation_values, right.imputation_values)
    for a, b in [(left.coefficients, right.coefficients), (left.intercepts, right.intercepts),
                 (left.scaler.mean_, right.scaler.mean_), (left.scaler.scale_, right.scaler.scale_),
                 (left.predictors_on_controls.coef_, right.predictors_on_controls.coef_),
                 (left.target_on_controls.coef_, right.target_on_controls.coef_)]:
        np.testing.assert_array_equal(a, b)
