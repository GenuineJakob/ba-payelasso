import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from partially_penalized_lasso import PartiallyPenalizedLassoCV


@pytest.fixture
def sample():
    rng = np.random.default_rng(42)
    controls = ["vol"] + [f"vol_lag_{lag}" for lag in range(1, 6)]
    ar = rng.normal(size=(180, 6))
    macro = np.column_stack(
        [2 * ar[:, 0] + rng.normal(size=180), 100 * rng.normal(size=180)]
    )
    X = pd.DataFrame(np.column_stack([ar, macro]), columns=controls + ["macro", "macro_large"])
    y = 3 + ar @ np.array([0.8, 0.3, -0.2, 0.1, 0.05, -0.05])
    y += macro @ np.array([0.4, -0.006]) + rng.normal(scale=0.05, size=180)
    return X, y, controls


@pytest.mark.parametrize("solver", ["sklearn", "lars"])
def test_large_penalty_reduces_to_ar6_with_free_intercept(sample, solver):
    X, y, controls = sample
    model = PartiallyPenalizedLassoCV(controls, np.array([100.0]), solver=solver).fit(X, y)
    benchmark = LinearRegression().fit(X[controls], y)
    np.testing.assert_array_equal(model.coef_[6:], 0.0)
    np.testing.assert_allclose(model.coef_[:6], benchmark.coef_, atol=1e-12)
    assert model.intercept_ == pytest.approx(benchmark.intercept_, abs=1e-12)
    np.testing.assert_allclose(model.predict(X), benchmark.predict(X[controls]), atol=1e-12)

    # Alpha selection must score the complete AR+Lasso prediction, not merely
    # compare validation targets to predictions of residualized target values.
    train, validation = next(TimeSeriesSplit(n_splits=5).split(X))
    fold_ar = LinearRegression().fit(X.iloc[train][controls], y[train])
    expected_mse = np.mean((y[validation] - fold_ar.predict(X.iloc[validation][controls])) ** 2)
    assert model.mse_path_[0, 0] == pytest.approx(expected_mse, abs=1e-12)


@pytest.mark.parametrize("solver", ["sklearn", "lars"])
def test_small_penalty_approaches_ols_in_original_units(sample, solver):
    X, y, controls = sample
    model = PartiallyPenalizedLassoCV(controls, np.array([1e-10]), tol=1e-10, solver=solver).fit(X, y)
    ols = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, ols.coef_, atol=1e-7)
    assert model.intercept_ == pytest.approx(ols.intercept_, abs=1e-7)
    np.testing.assert_allclose(model.predict(X), ols.predict(X), atol=1e-7)


def test_control_normal_equations_hold_under_nonzero_penalty(sample):
    X, y, controls = sample
    model = PartiallyPenalizedLassoCV(controls, np.array([0.1])).fit(X, y)
    errors = y - model.predict(X)
    assert np.any(model.coef_[6:] != 0)
    assert errors.mean() == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(X[controls].to_numpy().T @ errors / len(X), 0, atol=1e-12)
    # Named columns may be supplied in a different order at prediction time.
    np.testing.assert_allclose(model.predict(X[X.columns[::-1]]), model.predict(X))


def test_validation_changes_do_not_leak_into_training_projections_or_scaling(sample):
    X, y, controls = sample

    class RecordingModel(PartiallyPenalizedLassoCV):
        def _fit_path(self, X, y, alphas):
            path = super()._fit_path(X, y, alphas)
            self.recorded.append((X.index.to_numpy(), path))
            return path

    original = RecordingModel(controls, np.array([0.1, 0.01]))
    original.recorded = []
    original.fit(X, y)
    train, _ = next(TimeSeriesSplit(n_splits=5).split(X))
    changed_X, changed_y = X.copy(), y.copy()
    changed_X.iloc[len(train):] += 3
    changed_y[len(train):] += 10
    changed = RecordingModel(controls, np.array([0.1, 0.01]))
    changed.recorded = []
    changed.fit(changed_X, changed_y)

    first_rows, first = original.recorded[0]
    changed_rows, other = changed.recorded[0]
    np.testing.assert_array_equal(first_rows, train)
    np.testing.assert_array_equal(first_rows, changed_rows)
    for left, right in [
        (first.target_on_controls.coef_, other.target_on_controls.coef_),
        (first.target_on_controls.intercept_, other.target_on_controls.intercept_),
        (first.predictors_on_controls.coef_, other.predictors_on_controls.coef_),
        (first.predictors_on_controls.intercept_, other.predictors_on_controls.intercept_),
        (first.scaler.mean_, other.scaler.mean_),
        (first.scaler.scale_, other.scaler.scale_),
        (first.coefficients, other.coefficients),
        (first.intercepts, other.intercepts),
    ]:
        np.testing.assert_allclose(left, right, atol=0, rtol=0)
    assert not np.allclose(original.mse_path_[:, 0], changed.mse_path_[:, 0])


def test_redundant_or_constant_predictors_have_no_additional_signal(sample):
    X, y, controls = sample
    X = X[controls].copy()
    X["constant"] = 1.0
    X["duplicate_vol"] = 2 * X["vol"]
    model = PartiallyPenalizedLassoCV(controls, np.array([0.01])).fit(X, y)
    np.testing.assert_array_equal(model.coef_[6:], 0.0)
    assert np.isfinite(model.predict(X)).all()


def test_missing_control_is_rejected_instead_of_penalized(sample):
    X, y, controls = sample
    with pytest.raises(ValueError, match="Missing unpenalized controls"):
        PartiallyPenalizedLassoCV(controls, np.array([0.1])).fit(X.drop(columns=controls[0]), y)


def test_fold_local_imputation_cannot_use_validation_values(sample):
    X, y, controls = sample
    X = X.copy()
    X.loc[X.index[::5], "macro"] = np.nan

    class RecordingModel(PartiallyPenalizedLassoCV):
        def _fit_path(self, X, y, alphas):
            path = super()._fit_path(X, y, alphas)
            self.paths.append(path)
            return path

    train, _ = next(TimeSeriesSplit(n_splits=5).split(X))
    model = RecordingModel(controls, np.array([0.01, 0.1]), impute_missing=True)
    model.paths = []
    model.fit(X, y)
    altered = X.copy()
    altered.iloc[len(train):, altered.columns.get_loc("macro")] = 10000
    changed = RecordingModel(controls, np.array([0.01, 0.1]), impute_missing=True)
    changed.paths = []
    changed.fit(altered, y)
    expected = X.iloc[train]["macro"].median()
    assert model.paths[0].imputation_values["macro"] == pytest.approx(expected)
    pd.testing.assert_series_equal(model.paths[0].imputation_values, changed.paths[0].imputation_values)
    np.testing.assert_allclose(model.paths[0].coefficients, changed.paths[0].coefficients, atol=0, rtol=0)
    np.testing.assert_allclose(model.paths[0].scaler.scale_, changed.paths[0].scaler.scale_, atol=0, rtol=0)
    assert model.imputation_values_["macro"] == pytest.approx(X["macro"].median())


def test_imputation_handles_all_missing_features_without_imputing_controls(sample):
    X, y, controls = sample
    X = X.copy()
    X["all_missing"] = np.nan
    X.loc[X.index[::7], "macro"] = np.nan
    model = PartiallyPenalizedLassoCV(controls, np.array([0.1]), impute_missing=True).fit(X, y)
    assert model.imputation_values_["all_missing"] == 0
    assert model.coef_[-1] == 0
    assert np.isfinite(model.predict(X)).all()
    with pytest.raises(ValueError, match="imputation is enabled"):
        PartiallyPenalizedLassoCV(controls, np.array([0.1])).fit(X, y)
    bad = X.copy()
    bad.loc[0, controls[0]] = np.nan
    with pytest.raises(ValueError, match="controls and target must be finite"):
        PartiallyPenalizedLassoCV(controls, np.array([0.1]), impute_missing=True).fit(bad, y)
    with pytest.raises(ValueError, match="Prediction features must be finite"):
        model.predict(bad)


def test_lars_matches_coordinate_descent_for_the_same_penalized_objective(sample):
    X, y, controls = sample
    models = [PartiallyPenalizedLassoCV(controls, np.array([0.1]), tol=1e-9, solver=solver).fit(X, y)
              for solver in ["sklearn", "lars"]]
    np.testing.assert_allclose(models[0].coef_, models[1].coef_, atol=1e-6)
    np.testing.assert_allclose(models[0].predict(X), models[1].predict(X), atol=1e-6)
    for model in models:
        assert model.relative_dual_gap_ <= 1e-8
        assert model.selected_cv_relative_dual_gap_ <= 1e-8


def test_lars_target_conditioning_preserves_original_units(sample):
    X, y, controls = sample
    alphas = np.array([0.001, 0.01, 0.1])
    scale = 0.001
    original = PartiallyPenalizedLassoCV(controls, alphas, solver="lars").fit(X, y)
    rescaled = PartiallyPenalizedLassoCV(controls, alphas * scale, solver="lars").fit(X, y * scale)
    assert rescaled.alpha_ == pytest.approx(original.alpha_ * scale)
    np.testing.assert_allclose(rescaled.coef_, original.coef_ * scale, atol=1e-10)
    assert rescaled.intercept_ == pytest.approx(original.intercept_ * scale, abs=1e-10)
    np.testing.assert_allclose(rescaled.predict(X), original.predict(X) * scale, atol=1e-10)
    np.testing.assert_allclose(rescaled.mse_path_, original.mse_path_ * scale ** 2, atol=1e-12)
