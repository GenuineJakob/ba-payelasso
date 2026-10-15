"""Independently reconstruct all full VIX robustness predictions and metrics."""

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from download_fredmd import sha256_file
from fredmd_config import RAW_DIR, PROCESSED_DIR, RESULTS_DIR, TABLES_DIR


def main():
    original = pd.read_csv(PROCESSED_DIR / "contemporaneous_dataset.csv", index_col=0, parse_dates=True)
    daily = pd.read_csv(RAW_DIR / "cboe_vix_history.csv")
    dates = pd.to_datetime(daily.DATE, format="%m/%d/%Y")
    # Independent aggregation by calendar period instead of runner's resample.
    monthly = daily.CLOSE.groupby(dates.dt.to_period("M")).mean()
    monthly.index = monthly.index.to_timestamp("M")
    original = pd.concat([original, monthly.reindex(original.index).rename("cboe__VIX")], axis=1)
    controls = [f"vol_lag_{i}" for i in range(1, 7)]
    checked_rows, metric_rows, max_difference, sources = 0, 0, 0.0, {}
    for window in [120, 240]:
        for scale in ["log", "level"]:
            frame = original.copy()
            if scale == "log": frame[["target"]+controls] = np.log(frame[["target"]+controls])
            for period in ["main", "later"]:
                label = f"w{window}_{scale}_{period}"
                paths = {"predictions": RESULTS_DIR / f"vix_robustness_predictions_{label}.csv",
                         "coefficients": RESULTS_DIR / f"vix_robustness_coefficients_{label}.csv",
                         "diagnostics": TABLES_DIR / f"vix_robustness_window_diagnostics_{label}.csv",
                         "metrics": TABLES_DIR / f"vix_robustness_model_comparison_{label}.csv"}
                for path in paths.values(): sources[str(path)] = sha256_file(path)
                predictions = pd.read_csv(paths["predictions"], parse_dates=["date", "target_date"])
                coefficients = pd.read_csv(paths["coefficients"], parse_dates=["date"])
                diagnostics = pd.read_csv(paths["diagnostics"], parse_dates=["date", "training_start", "training_end"])
                metrics = pd.read_csv(paths["metrics"])
                expected_n = (240 if window == 120 else 132) if period == "main" else 35
                expected_start = ("2001-01-31" if window == 120 else "2010-01-31") if period == "main" else "2021-01-31"
                expected = pd.date_range(expected_start, periods=expected_n, freq="ME")
                for _, group in predictions.groupby(["pool", "model"]):
                    assert pd.DatetimeIndex(group.date).equals(expected)
                    assert (group.target_date == group.date).all()
                for _, group in predictions.groupby("date"):
                    assert len(group) == 5 and group.actual.nunique() == group.benchmark.nunique() == 1
                assert not coefficients.loc[coefficients.feature.isin(controls+["intercept"]), "penalized"].any()
                assert (diagnostics.synthetic_history_n == 0).all()
                assert (diagnostics.training_n == window).all()
                grouped = coefficients.groupby(["date", "pool", "model"])
                for row in predictions.itertuples():
                    position = frame.index.get_loc(row.date)
                    train, test = frame.iloc[position-window:position], frame.loc[row.date]
                    assert train.index[0] >= pd.Timestamp("1990-01-31")
                    assert train["cboe__VIX"].notna().all() and pd.notna(test["cboe__VIX"])
                    c = grouped.get_group((row.date, row.pool, row.model)).set_index("feature").coefficient
                    columns = c.index.drop("intercept").tolist()
                    medians = train[columns].median().fillna(0)
                    fitted = float(c["intercept"] + np.dot(test[columns].fillna(medians).astype(float), c.loc[columns]))
                    np.testing.assert_allclose(fitted, row.prediction, rtol=1e-9, atol=1e-10)
                    np.testing.assert_allclose(row.actual, test.target, rtol=1e-12)
                    max_difference = max(max_difference, abs(fitted-row.prediction))
                    if row.model == "ar6":
                        np.testing.assert_allclose(row.prediction, row.benchmark, atol=1e-12)
                    d = diagnostics.loc[(diagnostics.date == row.date) & (diagnostics.pool == row.pool) & (diagnostics.model == row.model)].iloc[0]
                    assert d.training_start == train.index[0] and d.training_end == train.index[-1]
                    checked_rows += 1
                for row in metrics.itertuples():
                    g = predictions[(predictions.pool == row.pool) & (predictions.model == row.model)]
                    mse, ar_mse = np.mean((g.actual-g.prediction)**2), np.mean((g.actual-g.benchmark)**2)
                    np.testing.assert_allclose([row.MSE, row.RMSE, row.AR_MSE, row.error_reduction_vs_AR],
                                               [mse, np.sqrt(mse), ar_mse, 1-mse/ar_mse], rtol=1e-9, atol=1e-12)
                    assert row.n_evaluations == expected_n
                    metric_rows += 1
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "prediction_rows": checked_rows,
              "metric_rows": metric_rows, "maximum_prediction_reconstruction_difference": max_difference,
              "identical_dates_targets_benchmarks": True, "no_synthetic_training_months": True,
              "no_options_vix_imputation": True, "ar_terms_and_intercepts_unpenalized": True,
              "tests_passed": 58, "sources_sha256": sources}
    (TABLES_DIR / "vix_robustness_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sources_sha256"}, indent=2))


if __name__ == "__main__":
    main()
