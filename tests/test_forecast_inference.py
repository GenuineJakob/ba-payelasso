"""Check inference against independent calculations and adverse input cases."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from forecast_inference import (hac_mean_test, holm_adjust, paired_block_interval,
                               circular_block_indices, validate_monthly_pairs,
                               load_comparisons, run)


def test_hac_matches_independent_bartlett_covariance_matrix():
    reference = np.array([2., 3., 1., 4., 2., 1., 3., 5.])
    candidate = np.array([1., 2., 2., 2., 3., 1., 1., 2.])
    d = reference - candidate
    residual = d - d.mean()
    lag = 3
    distance = np.abs(np.arange(len(d))[:, None] - np.arange(len(d)))
    weights = np.maximum(0, 1 - distance / (lag + 1))
    lrv = residual @ weights @ residual / len(d)
    expected_se = np.sqrt(lrv / len(d))
    result = hac_mean_test(reference, candidate, lags=lag)
    assert result["test_status"] == "ok"
    assert result["hac_long_run_variance"] == pytest.approx(lrv)
    assert result["hac_standard_error"] == pytest.approx(expected_se)
    assert result["hac_z"] == pytest.approx(d.mean()/expected_se)
    assert result["hac_p_two_sided"] == pytest.approx(2*norm.sf(abs(d.mean()/expected_se)))


def test_hac_scale_invariance_and_reversing_comparison():
    reference = np.array([4., 3., 2., 6., 4., 2.])
    candidate = np.array([1., 5., 1., 2., 3., 2.])
    base = hac_mean_test(reference, candidate, lags=2)
    scaled = hac_mean_test(reference*1e-12, candidate*1e-12, lags=2)
    reversed_pair = hac_mean_test(candidate, reference, lags=2)
    assert scaled["hac_z"] == pytest.approx(base["hac_z"])
    assert reversed_pair["hac_z"] == pytest.approx(-base["hac_z"])
    assert reversed_pair["hac_p_two_sided"] == pytest.approx(base["hac_p_two_sided"])
    assert reversed_pair["hac_difference_ci_low"] == pytest.approx(-base["hac_difference_ci_high"])


def test_serial_dependence_changes_hac_uncertainty():
    d = np.repeat([0.1, 0.5, 0.2, 0.8], 30)
    iid = hac_mean_test(1 + d, np.ones(len(d)), lags=0)
    dependent = hac_mean_test(1 + d, np.ones(len(d)), lags=6)
    assert dependent["hac_standard_error"] > iid["hac_standard_error"] * 2


@pytest.mark.parametrize("candidate,status", [([1, 2, 3], "numerically_identical_losses"),
                                              ([0, 1, 2], "degenerate_loss_variance")])
def test_degenerate_losses_do_not_generate_a_pvalue(candidate, status):
    result = hac_mean_test([1, 2, 3], candidate, lags=1)
    assert result["test_status"] == status
    assert np.isnan(result["hac_p_two_sided"])


def test_holm_known_values_and_undefined_tests():
    np.testing.assert_allclose(holm_adjust([0.04, 0.01, 0.03]), [0.06, 0.03, 0.06])
    result = holm_adjust([0.01, np.nan, 0.02])
    np.testing.assert_allclose(result[[0, 2]], [0.03, 0.04])
    assert np.isnan(result[1])


def test_circular_indices_preserve_consecutive_blocks_and_wrap():
    indices = circular_block_indices(11, 4, 100, np.random.default_rng(1))
    assert indices.shape == (100, 11)
    for start, stop in [(0, 4), (4, 8), (8, 11)]:
        assert np.all(np.diff(indices[:, start:stop], axis=1) % 11 == 1)
    assert np.any(np.diff(indices[:, :4], axis=1) == -10)


def test_bootstrap_keeps_pairing_and_recomputes_ratio_denominator():
    reference = np.linspace(0.1, 4, 36)**2
    candidate = 0.75*reference
    result = paired_block_interval(reference, candidate, block_length=6, resamples=1000, seed=11)
    assert result["bootstrap_reduction_ci_low"] == pytest.approx(0.25)
    assert result["bootstrap_reduction_ci_high"] == pytest.approx(0.25)
    # Incorrect independent date draws or a fixed original denominator would
    # produce a nonconstant ratio interval for these perfectly proportional losses.
    identical = paired_block_interval(reference, reference, resamples=1000, seed=11)
    assert identical["bootstrap_reduction_ci_low"] == 0
    assert identical["bootstrap_reduction_ci_high"] == 0
    assert result == paired_block_interval(reference, candidate, block_length=6, resamples=1000, seed=11)


@pytest.mark.parametrize("reference,candidate", [([1, np.nan], [1, 2]), ([1, 2], [1]),
                                                ([0, 0], [1, 2]), ([-1, 2], [1, 2])])
def test_invalid_losses_fail(reference, candidate):
    with pytest.raises(ValueError):
        hac_mean_test(reference, candidate)


def monthly_pairs():
    dates = pd.date_range("2001-01-31", periods=4, freq="ME")
    return pd.DataFrame({"date": dates, "target_date": dates + pd.offsets.MonthEnd(1),
                         "actual": [1., 2., 3., 4.], "reference": [1.1, 2.2, 3.3, 4.4],
                         "candidate": [1.2, 2.1, 3.4, 4.3]})


@pytest.mark.parametrize("problem", ["duplicate", "gap", "wrong_target", "nan", "not_month_end"])
def test_pair_validation_rejects_unsafe_alignment(problem):
    pairs = monthly_pairs()
    if problem == "duplicate":
        pairs.loc[1, "date"] = pairs.loc[0, "date"]
    elif problem == "gap":
        pairs = pairs.drop(1)
    elif problem == "wrong_target":
        pairs.loc[1, "target_date"] = pairs.loc[1, "date"]
    elif problem == "nan":
        pairs.loc[1, "candidate"] = np.nan
    else:
        pairs.loc[0, "date"] -= pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        validate_monthly_pairs(pairs)


def test_pair_validation_sorts_and_computes_losses():
    result = validate_monthly_pairs(monthly_pairs().iloc[::-1])
    assert result.date.is_monotonic_increasing
    np.testing.assert_allclose(result.reference_loss, [0.01, 0.04, 0.09, 0.16])


def write_fixture(root):
    (root / "output/results").mkdir(parents=True)
    (root / "output/tables").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src/forecast_inference.py").write_text("# fixture source\n")
    dates = pd.date_range("2001-01-31", periods=36, freq="ME")
    actual = 0.2 + np.arange(36)/1000
    ar = actual + np.linspace(0.01, 0.08, 36)
    candidate = actual + 0.9*(ar-actual)
    ols = actual + 1.5*(ar-actual)
    for design in ["rolling", "fixed"]:
        lasso = candidate if design == "rolling" else ar
        frame = pd.DataFrame({"date": dates, "actual": actual, "benchmark": ar, "lasso": lasso})
        frame.to_csv(root / f"output/results/{design}_predictions.csv", index=False)
        pd.DataFrame({"model": ["benchmark", "lasso"],
                      "MSE": [np.mean((actual-ar)**2), np.mean((actual-lasso)**2)]}).to_csv(
                          root / f"output/tables/{design}_model_comparison.csv", index=False)
    for period in ["main", "holdout"]:
        parts, rows = [], []
        for model, prediction in [("ar6_benchmark", ar), ("lasso", candidate), ("ols", ols)]:
            parts.append(pd.DataFrame({"date": dates, "target_date": dates + pd.offsets.MonthEnd(1),
                                       "actual": actual, "benchmark": ar, "prediction": prediction,
                                       "model": model, "pool": "fredmd_macro", "period": period}))
            rows.append({"pool": "fredmd_macro", "model": model, "MSE": np.mean((actual-prediction)**2)})
        pd.concat(parts).to_csv(root / f"output/results/fredmd_predictions_{period}.csv", index=False)
        pd.DataFrame(rows).to_csv(root / f"output/tables/fredmd_model_comparison_{period}.csv", index=False)
        (root / f"output/results/fredmd_run_manifest_{period}.json").write_text(json.dumps(
            {"smoke_run": False, "window_size": 240, "n_splits": 5, "n_forecasts": 36}))
        pd.DataFrame({"pool": ["fredmd_macro"]*36, "training_n": [240]*36}).to_csv(
            root / f"output/tables/fredmd_window_diagnostics_{period}.csv", index=False)


def test_end_to_end_preserves_inputs_and_adjusts_only_selected_families(tmp_path):
    write_fixture(tmp_path)
    inputs = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    summary, manifest = run(tmp_path, resamples=1000, seed=12)
    assert len(summary) == 6
    assert (summary.role == "primary").sum() == 3
    assert (summary.role == "retrospective_robustness").sum() == 2
    diagnostic = summary.loc[summary.role == "diagnostic"].iloc[0]
    assert diagnostic.test_status == "numerically_identical_losses"
    assert np.isnan(diagnostic.hac_p_holm)
    assert diagnostic.bootstrap_reduction_ci_low == 0
    primary = summary.loc[summary.role == "primary"]
    np.testing.assert_allclose(primary.hac_p_holm, holm_adjust(primary.hac_p_two_sided))
    for path, digest in inputs.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    for relative, digest in manifest["outputs_sha256"].items():
        assert hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() == digest
    previous_summary = (tmp_path / "output/tables/forecast_inference_summary.csv").read_bytes()
    run(tmp_path, resamples=1000, seed=12)
    assert (tmp_path / "output/tables/forecast_inference_summary.csv").read_bytes() == previous_summary


def test_loader_rejects_model_target_mismatch(tmp_path):
    write_fixture(tmp_path)
    path = tmp_path / "output/results/fredmd_predictions_main.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.model == "ols", "actual"] += 0.01
    frame.to_csv(path, index=False)
    with pytest.raises(AssertionError):
        load_comparisons(tmp_path, {})


def test_saved_results_reconcile_with_loss_files_and_independent_hac():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "output/tables/forecast_inference_manifest.json"
    if not manifest_path.exists():
        pytest.skip("No saved inference run in this test checkout.")
    manifest = json.loads(manifest_path.read_text())
    for relative, digest in {**manifest["sources_sha256"], **manifest["outputs_sha256"],
                             **manifest["unchanged_prediction_files_sha256"]}.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
    summary = pd.read_csv(root / "output/tables/forecast_inference_summary.csv")
    losses = pd.read_csv(root / "output/results/forecast_inference_losses.csv")
    for _, row in summary.iterrows():
        pair = losses.loc[losses.comparison_id == row.comparison_id]
        assert len(pair) == row.n
        ref = (pair.actual - pair.reference)**2
        cand = (pair.actual - pair.candidate)**2
        np.testing.assert_allclose(ref, pair.reference_loss, atol=1e-16)
        np.testing.assert_allclose(cand, pair.candidate_loss, atol=1e-16)
        assert ref.mean() == pytest.approx(row.reference_mse)
        assert cand.mean() == pytest.approx(row.candidate_mse)
        assert 1 - cand.mean()/ref.mean() == pytest.approx(row.reduction, abs=1e-14)
        if row.test_status != "ok":
            assert pd.isna(row.hac_p_two_sided)
            continue
        differential = (ref - cand).to_numpy()
        centered = differential - differential.mean()
        distance = np.abs(np.arange(len(pair))[:, None] - np.arange(len(pair)))
        weights = np.maximum(0, 1 - distance/(row.hac_lags + 1))
        se = np.sqrt(centered @ weights @ centered / len(pair)**2)
        assert se == pytest.approx(row.hac_standard_error, rel=1e-10)
        assert 2*norm.sf(abs(differential.mean()/se)) == pytest.approx(row.hac_p_two_sided, rel=1e-10)
