"""Options-only history, matched windows and output checks."""

import numpy as np
import pandas as pd
import pytest

from train_vix_robustness import (
    VIX, OPTIONS_VIX, CONTROLS, monthly_vix, build_matched_data, matched_dates, run,
)
from download_vix import cache_history


def fixture_data():
    rng = np.random.default_rng(82)
    dates = pd.date_range("1980-01-31", "2023-11-30", freq="ME")
    frame = pd.DataFrame({"target": np.exp(rng.normal(-2, .3, len(dates))),
                          "target_date": dates, "macro": rng.normal(size=len(dates)),
                          VIX: rng.uniform(10, 40, len(dates))}, index=dates)
    for i, column in enumerate(CONTROLS, 1):
        frame[column] = frame.target.shift(i)
    options = frame[VIX].loc["1990":].rename(OPTIONS_VIX)
    pools = {"fredmd_full": ["macro", VIX], "fredmd_no_iv": ["macro"]}
    return build_matched_data(frame, pools, options)


def test_monthly_close_and_future_independence():
    daily = pd.DataFrame({"DATE": ["01/02/1990", "01/03/1990", "03/01/1990"],
                          "CLOSE": [10, 20, 100], "OPEN": [900, 900, 900]})
    result = monthly_vix(daily)
    assert result.iloc[0] == 15
    assert np.isnan(result.iloc[1])  # Keep February; never interpolate.
    changed = daily.copy()
    changed.loc[2, "CLOSE"] = 200
    pd.testing.assert_series_equal(result.iloc[:2], monthly_vix(changed).iloc[:2])


@pytest.mark.parametrize("change", ["duplicate", "negative", "missing", "pre1990"])
def test_reject_invalid_options_history(change):
    daily = pd.DataFrame({"DATE": ["01/02/1990", "01/03/1990"], "CLOSE": [10, 20]})
    if change == "duplicate": daily.loc[1, "DATE"] = daily.loc[0, "DATE"]
    if change == "negative": daily.loc[0, "CLOSE"] = -1
    if change == "missing": daily.loc[0, "CLOSE"] = np.nan
    if change == "pre1990": daily.loc[0, "DATE"] = "12/29/1989"
    with pytest.raises(ValueError): monthly_vix(daily)


def test_matched_pools_and_training_history():
    data, pools = fixture_data()
    assert set(pools["options_only"]) - set(pools["no_volatility_proxy"]) == {OPTIONS_VIX}
    assert set(pools["extended_fredmd"]) - set(pools["no_volatility_proxy"]) == {VIX}
    assert data.loc[:"1989", OPTIONS_VIX].isna().all()
    dates = matched_dates(data, 240, "main")
    assert len(dates) == 132 and dates[0] == pd.Timestamp("2010-01-31")
    assert len(matched_dates(data, 120, "main")) == 240
    altered = data.copy()
    altered.loc["2005-01-31", OPTIONS_VIX] = np.nan
    with pytest.raises(ValueError, match="no imputation"): matched_dates(altered, 240, "main")


def test_cache_frozen_and_checksum_checked(tmp_path):
    csv = tmp_path / "download.csv"
    csv.write_text("DATE,CLOSE\n01/02/1990,10\n01/03/1990,20\n")
    path = cache_history(tmp_path / "raw", csv)
    csv.write_text("DATE,CLOSE\n01/02/1990,100\n01/03/1990,200\n")
    assert cache_history(tmp_path / "raw", csv).read_text().endswith("01/03/1990,20\n")
    path.write_text(path.read_text()+"01/04/1990,30\n")
    with pytest.raises(ValueError, match="checksum"): cache_history(tmp_path / "raw")


def test_short_run_matching_metrics_and_unpenalized_controls(tmp_path):
    data, pools = fixture_data()
    data.loc["2005-01-31", "macro"] = np.nan
    results, tables = tmp_path / "results", tmp_path / "tables"
    metrics = run(data, pools, {}, 240, "log", "main", max_forecasts=2,
                  results_dir=results, tables_dir=tables)
    p = pd.read_csv(results / "vix_robustness_predictions_w240_log_main_smoke.csv")
    for _, group in p.groupby("date"):
        assert len(group) == 5
        assert group.actual.nunique() == group.benchmark.nunique() == 1
    for _, row in metrics.iterrows():
        g = p[(p.pool == row.pool) & (p.model == row.model)]
        mse = np.mean((g.actual-g.prediction)**2)
        assert row.MSE == pytest.approx(mse)
        assert row.error_reduction_vs_AR == pytest.approx(1-mse/np.mean((g.actual-g.benchmark)**2))
    c = pd.read_csv(results / "vix_robustness_coefficients_w240_log_main_smoke.csv")
    assert not c.loc[c.feature.isin(CONTROLS+["intercept"]), "penalized"].any()
    d = pd.read_csv(tables / "vix_robustness_window_diagnostics_w240_log_main_smoke.csv")
    assert (d.synthetic_history_n == 0).all()
    assert (d.training_n == 240).all()
