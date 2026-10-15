"""Short real-data forecasts; all generated outputs stay in pytest's temp dir."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import paye_monthly_utils as paye
import train_models as original
import train_paye_monthly_batch_experiments as batch
import train_paye_monthly_individual as individual
import train_paye_monthly_kitchen_sink as kitchen_sink


def test_original_fixed_split_and_short_rolling_forecast():
    df = original.load_dataset()
    assert len(original.BENCHMARK_PREDICTORS) == 6
    assert df[original.FULL_PREDICTORS + [original.TARGET]].notna().all().all()
    raw = pd.read_csv(original.DATA_PATH, index_col=0, parse_dates=True).sort_index()
    for lag in range(1, 6):
        pd.testing.assert_series_equal(
            df[f"vol_lag_{lag}"], raw["volatility"].shift(lag).loc[df.index], check_names=False
        )

    for predictions, metrics, lasso_coefficients, ols_coefficients in [
        original.fixed_split_forecast(df),
        original.rolling_window_forecast(
            df, end_date=df.index[original.ROLLING_WINDOW_SIZE + 2]
        ),
    ]:
        assert np.isfinite(predictions[["actual", "benchmark", "ols", "lasso"]]).all().all()
        assert np.isfinite(metrics[["MSE", "RMSE"]]).all().all()
        for coefficients in [lasso_coefficients, ols_coefficients]:
            names = coefficients.index if isinstance(coefficients, pd.Series) else coefficients.columns
            assert set(["intercept"] + original.FULL_PREDICTORS).issubset(names)
            assert np.isfinite(coefficients.to_numpy()).all()
        if "selected_variables" in predictions:
            for selected in predictions["selected_variables"]:
                assert not set(selected.split(",")).intersection(original.BENCHMARK_PREDICTORS + ["intercept"])

    coefficients = pd.Series(1.0, index=["intercept"] + original.FULL_PREDICTORS)
    assert set(original.selected_lasso_coefficients(coefficients).index).isdisjoint(
        original.BENCHMARK_PREDICTORS + ["intercept"]
    )


@pytest.fixture
def temporary_paye_outputs(monkeypatch, tmp_path):
    results = tmp_path / "results"
    tables = tmp_path / "tables"
    results.mkdir()
    tables.mkdir()
    for module in [paye, individual, kitchen_sink, batch]:
        monkeypatch.setattr(module, "RESULTS_DIR", results)
        monkeypatch.setattr(module, "TABLES_DIR", tables)
    monkeypatch.setattr(paye, "AR6_BENCHMARK_PATH", results / "benchmark.csv")

    # Keep actual 240-month training windows and the full 200-alpha grid,
    # but produce only three forecast dates per model.
    data = paye.load_monthly_data().loc[:"2001-03-31"]
    monkeypatch.setattr(individual, "load_monthly_data", lambda: data.copy())
    monkeypatch.setattr(kitchen_sink, "load_monthly_data", lambda: data.copy())
    return data, results, tables


def test_individual_and_kitchen_sink_scripts_export_complete_coefficients(temporary_paye_outputs):
    data, results, tables = temporary_paye_outputs
    available = paye.available_paye_predictors(data)
    individual.main()
    kitchen_sink.main()
    for name in ["individual", "kitchen_sink"]:
        lasso = pd.read_csv(results / f"paye_monthly_{name}_lasso_coefficients.csv")
        ols = pd.read_csv(results / f"paye_monthly_{name}_ols_coefficients.csv")
        for coefficients in [lasso, ols]:
            assert set(["intercept"] + paye.AR6_PREDICTORS).issubset(coefficients.columns)
            # Individual-model exports concatenate different predictor sets;
            # NaNs in columns belonging to other models are intentional.
            if name == "individual":
                for predictor, column in available.items():
                    group = coefficients[coefficients["paye_predictor"] == predictor]
                    columns = ["intercept"] + paye.AR6_PREDICTORS + [column]
                    assert len(group) == 3
                    assert np.isfinite(group[columns]).all().all()
            else:
                assert np.isfinite(coefficients.select_dtypes(include="number")).all().all()
        assert "alpha" in lasso.columns
        assert np.isfinite(lasso["alpha"]).all()
        predictions = pd.read_csv(results / f"paye_monthly_{name}_predictions.csv")
        assert np.isfinite(predictions[["actual", "benchmark", "ols", "lasso"]]).all().all()
        assert (tables / f"paye_monthly_{name}_model_comparison.csv").exists()


def test_batch_experiment_and_selection_report(temporary_paye_outputs, monkeypatch):
    import evaluate_paye_monthly_outputs as evaluation

    data, results, _ = temporary_paye_outputs
    available = paye.available_paye_predictors(data)
    sample = batch.common_experiment_sample(data, available)
    metrics, predictions = batch.run_experiment(
        df=sample,
        experiment_name="smoke",
        block_name="all_available",
        block_predictor_names=batch.PREDICTOR_BLOCKS["all_available"],
        window_size=240,
        forecast_start="2001-01-31",
        forecast_end="2001-03-31",
        alpha_grid_name="moderate_1e-4_1e0",
        alphas=batch.ALPHA_GRIDS["moderate_1e-4_1e0"],
        available=available,
        include_baseline_models=True,
    )
    assert metrics
    assert np.isfinite(predictions[["actual", "lasso"]]).all().all()
    monkeypatch.setattr(evaluation, "RESULTS_DIR", results)
    selection = evaluation.lasso_selection_frequency()
    assert not selection.empty
    assert set(selection["coefficient"]).isdisjoint(paye.AR6_PREDICTORS + ["intercept", "alpha"])


def test_conceptual_example_uses_shared_estimator(monkeypatch):
    data = paye.load_monthly_data().loc[:"2001-02-28"].tail(247).copy()
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: data.copy())
    namespace = runpy.run_path(str(Path(paye.__file__).with_name("conceptual_ar6_lasso_example.py")))
    assert len(namespace["ar_variables"]) == 6
    assert len(namespace["results"]) == 2
    assert np.isfinite(namespace["results"].to_numpy()).all()
