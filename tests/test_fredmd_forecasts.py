import numpy as np
import pandas as pd
import pytest

from fredmd_config import AR_CONTROLS, DATASET_NAME, POOL_NAMES
from partially_penalized_lasso import training_medians
from test_fredmd_data import frozen_experiment, synthetic_sources
import train_fredmd as runner


def test_short_main_and_holdout_exports_use_identical_dates_and_reconcile(frozen_experiment, tmp_path, monkeypatch):
    processed, _, _ = frozen_experiment
    data, metadata, manifest = runner.load_experiment(processed)
    # Full 200-alpha real-data paths are checked separately by CLI smoke runs.
    monkeypatch.setattr(runner, "ALPHAS", np.array([0.01, 0.1]))
    results, tables = tmp_path / "results", tmp_path / "tables"
    sentinel = results / "previous_result.csv"
    results.mkdir()
    sentinel.write_text("Existing results must be preserved.\n")
    for period in ["main", "holdout"]:
        predictions, metrics = runner.run_period(data, metadata, manifest, period, results, tables, max_forecasts=3)
        assert len(predictions) == 3 * 4 * 3
        assert len(metrics) == 4 * 3
        assert np.isfinite(predictions[["actual", "prediction", "benchmark"]]).all().all()
        dates = predictions[predictions.pool == POOL_NAMES[0]].date.unique()
        for pool in POOL_NAMES:
            for model in ["ar6_benchmark", "ols", "lasso"]:
                group = predictions[(predictions.pool == pool) & (predictions.model == model)]
                np.testing.assert_array_equal(group.date, dates)
                np.testing.assert_array_equal(group.target_date, group.date + pd.offsets.MonthEnd(1))
                row = metrics[(metrics.pool == pool) & (metrics.model == model)].iloc[0]
                mse = ((group.actual - group.prediction) ** 2).mean()
                base_mse = ((group.actual - group.benchmark) ** 2).mean()
                assert row.MSE == pytest.approx(mse)
                assert row.RMSE == pytest.approx(np.sqrt(mse))
                assert row.OOS_R2 == pytest.approx(1 - mse / base_mse)
                assert row.n_ar_terms == 6
                assert row.pool_size == len(manifest["pools"][pool])
                assert row.n_predictors == (0 if model == "ar6_benchmark" else row.pool_size)
        label = period + "_smoke"
        coefficients = pd.read_csv(results / f"fredmd_coefficients_{label}.csv", parse_dates=["date"])
        for _, row in coefficients.iterrows():
            features = manifest["pools"][row.pool]
            predictors = AR_CONTROLS + features
            position = data.index.get_loc(row.date)
            training = data.iloc[position - 240:position]
            values = data.loc[row.date, predictors].fillna(training_medians(training[features]))
            reconstructed = row.intercept + values @ row[predictors].astype(float)
            saved = predictions[(predictions.pool == row.pool) & (predictions.model == row.model) & (predictions.date == row.date)]
            assert reconstructed == pytest.approx(saved.prediction.iloc[0], abs=1e-10)
        windows = pd.read_csv(tables / f"fredmd_window_diagnostics_{label}.csv")
        assert (windows.training_n == 240).all()
        assert (windows.ols_rank <= windows.ols_fitted_columns).all()
        assert (windows.lasso_convergence_warnings == 0).all()
        assert (windows.lasso_relative_dual_gap <= 1.1e-6).all()
        assert (windows.lasso_selected_cv_relative_dual_gap <= 1.1e-6).all()
        assert (windows.lasso_max_cv_relative_dual_gap <= 1.1e-6).all()
        selection = pd.read_csv(tables / f"fredmd_selection_frequency_{label}.csv")
        assert set(selection.feature).isdisjoint(AR_CONTROLS + ["intercept"])
        assert selection.selected_share.between(0, 1).all()
    missing = pd.read_csv(results / "fredmd_imputation_values_main_smoke.csv")
    assert ((missing.feature == "fredmd__RPI") & missing.forecast_imputed).any()
    assert sentinel.read_text() == "Existing results must be preserved.\n"
    assert not (tables / "fredmd_model_comparison_main.csv").exists()


def test_processed_data_cannot_be_changed_after_freezing(frozen_experiment):
    processed, _, _ = frozen_experiment
    with (processed / DATASET_NAME).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.load_experiment(processed)


def test_ols_imputes_from_training_and_keeps_empty_predictors_at_zero():
    rng = np.random.default_rng(91)
    train = pd.DataFrame(rng.normal(size=(80, 6)), columns=AR_CONTROLS)
    train["macro"] = rng.normal(size=80)
    train.loc[train.index[::4], "macro"] = np.nan
    train["empty"] = np.nan
    train["target"] = 2 + 0.7 * train.vol + 0.3 * train["macro"].fillna(train["macro"].median())
    test = train.iloc[[0]].copy()
    test["macro"] = np.nan
    prediction, coefficients, diagnostics = runner.fit_ols_window(train, test, ["macro", "empty"])
    expected = 2 + 0.7 * test.vol.iloc[0] + 0.3 * train["macro"].median()
    assert prediction == pytest.approx(expected, abs=1e-10)
    assert coefficients["empty"] == 0
    assert diagnostics["inactive_predictors"] == 1
