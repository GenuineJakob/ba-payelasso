"""Guard against scale-dependent dense OLS truncation and preprocessing leakage."""
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import lstsq
from stable_ols import TrainingScaledOLS, software_versions


def test_independent_small_scale_direction_is_not_dropped():
    rng = np.random.default_rng(88)
    normalized = rng.normal(size=(240, 3))
    X = pd.DataFrame(normalized * [1e9, 1, 1e-8], columns=["large", "normal", "small"])
    y = 4 + normalized @ [2., -.5, 3.]
    fit = TrainingScaledOLS().fit(X, y)
    assert fit.rank_ == 3
    assert fit.raw_rank_1e6_ == 1
    np.testing.assert_allclose(fit.predict(X), y, atol=1e-12)
    np.testing.assert_allclose(fit.coef_ * [1e9, 1, 1e-8], [2, -.5, 3], atol=1e-12)
    assert fit.intercept_ == pytest.approx(4, abs=1e-12)
    assert fit.rcond_ == np.finfo(float).eps * 240


def test_prediction_invariant_to_units_and_exports_in_original_units():
    rng = np.random.default_rng(34)
    train = pd.DataFrame(rng.normal(size=(80, 8)), columns=list("abcdefgh"))
    test = pd.DataFrame(rng.normal(size=(7, 8)), columns=train.columns)
    y = 2 + train.to_numpy() @ np.arange(8) + rng.normal(size=80) / 10
    original = TrainingScaledOLS().fit(train, y)
    factors = np.logspace(-10, 10, 8)
    changed = TrainingScaledOLS().fit(train * factors, y)
    np.testing.assert_allclose(original.predict(test), changed.predict(test * factors), atol=1e-11)
    np.testing.assert_allclose(original.coef_, changed.coef_ * factors, atol=1e-11)
    np.testing.assert_allclose(original.predict(test), test.to_numpy() @ original.coef_ + original.intercept_)
    np.testing.assert_allclose(original.predict(test[train.columns[::-1]]), original.predict(test))


def test_scaler_parameters_use_training_only():
    rng = np.random.default_rng(55)
    train = pd.DataFrame(rng.normal(size=(80, 4)))
    y = rng.normal(size=80)
    fit = TrainingScaledOLS().fit(train, y)
    means, scales, coefs = fit.mean_.copy(), fit.scale_.copy(), fit.coef_.copy()
    fit.predict(train.iloc[[0]] * 1e10)
    np.testing.assert_array_equal(fit.mean_, means)
    np.testing.assert_array_equal(fit.scale_, scales)
    np.testing.assert_array_equal(fit.coef_, coefs)
    np.testing.assert_allclose(means, train.mean(), atol=1e-14)
    np.testing.assert_allclose(scales, train.std(ddof=0), atol=1e-14)


def test_exact_dependence_is_reported_at_the_explicit_cutoff():
    x = np.arange(100, dtype=float)
    X = pd.DataFrame({"x": x, "duplicate": 2*x, "constant": 1.})
    fit = TrainingScaledOLS().fit(X, 3*x + 2)
    assert fit.rank_ == 1
    assert fit.diagnostics()["ols_rank_deficient"]
    assert fit.coef_[2] == 0
    np.testing.assert_allclose(fit.predict(X), 3*x + 2, atol=1e-12)


def test_matches_independently_standardized_explicit_scipy_ols():
    rng = np.random.default_rng(90)
    X = rng.normal(size=(80, 20)) * np.logspace(-5, 5, 20)
    y = rng.normal(size=80)
    means, scales = X.mean(axis=0), X.std(axis=0)
    z = (X - means) / scales
    tolerance = np.finfo(float).eps * max(z.shape)
    independent_beta, _, independent_rank, _ = lstsq(z, y - y.mean(), cond=tolerance)
    fit = TrainingScaledOLS().fit(X, y)
    np.testing.assert_allclose(fit.predict(X), z @ independent_beta + y.mean(), atol=1e-12)
    assert fit.rank_ == independent_rank
    assert set(software_versions()) >= {"python", "numpy", "pandas", "scipy", "scikit-learn"}
