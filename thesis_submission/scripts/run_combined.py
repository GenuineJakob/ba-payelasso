"""Run the 108-predictor thesis extension using package-local frozen inputs.

Called by run_thesis.py; paths may also be supplied explicitly.
"""
import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

PACKAGE = Path(__file__).resolve().parent.parent
PROJECT = PACKAGE / "scripts"
FROZEN = PACKAGE / "results" / "reference" / "monthly"
sys.path.insert(0, str(PROJECT / "current_scope"))
import forecast
import models
import evaluate
import settings
sys.path.append(str(PROJECT / "legacy"))
from partially_penalized_lasso import PartiallyPenalizedLassoCV
from stable_ols import TrainingScaledOLS


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_inputs():
    manifest = json.loads((FROZEN / "data_manifest.json").read_text())
    previous = json.loads((FROZEN / "run_all.json").read_text())
    paths = [FROZEN / name for name in manifest["artifacts"]]
    paths += [FROZEN / name for name in ["predictions.csv", "metrics.csv", "run_all.json", "data_manifest.json"]]
    paths += list((PROJECT / "current_scope").glob("*.py"))
    paths += [PROJECT / "legacy" / name for name in ["partially_penalized_lasso.py", "stable_ols.py"]]
    for name, expected in manifest["artifacts"].items():
        if digest(FROZEN / name) != expected:
            raise ValueError(f"Frozen thesis input changed: {name}")
    for name, expected in previous["script_sha256"].items():
        if name == "run.py":  # The command-line wrapper is not used by this extra run.
            continue
        if digest(PROJECT / "current_scope" / name) != expected:
            raise ValueError(f"Original thesis method changed: {name}")
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]:
        if version(name) != previous["software"][name]:
            raise ValueError(f"Software differs from the thesis run: {name}")
    base = pd.read_csv(FROZEN / "base.csv", index_col=0, parse_dates=True)
    expanded = pd.read_csv(FROZEN / "expanded.csv", index_col=0, parse_dates=True)
    pools = json.loads((FROZEN / "pools.json").read_text())
    # Keep every baseline column, then add the macro pool to create a nested expansion.
    combined = pools["baseline"] + pools["macro"]
    if len(pools["baseline"]) != 9 or len(pools["macro"]) != 99 or len(set(combined)) != 108:
        raise ValueError("The combined design requires nine baseline and 99 distinct macro columns.")
    if not set(combined + settings.AR + ["target"]).issubset(expanded):
        raise ValueError("Combined predictors are absent from the frozen data.")
    return base, expanded, pools, combined, {str(p): digest(p) for p in paths}


def validate_estimators(expanded, combined):
    """Independent legacy implementation checks on two actual forecast origins."""
    checks = []
    columns = settings.AR + combined
    with threadpool_limits(limits=1):
        for date in ["2001-01-31", "2021-01-31"]:
            i = expanded.index.get_loc(pd.Timestamp(date))
            train, test = expanded.iloc[i-240:i], expanded.iloc[[i]]
            simple_lasso = models.lasso(train[columns], train.target, solver="lars")
            legacy_lasso = PartiallyPenalizedLassoCV(
                controls=settings.AR, alphas=settings.ALPHAS,
                impute_missing=True, solver="lars").fit(train[columns], train.target)
            simple_ols = models.ols(train[columns], train.target, scaled=True)
            medians = train[columns].median().fillna(0)
            legacy_ols = TrainingScaledOLS().fit(train[columns].fillna(medians), train.target)
            lasso_diff = float(abs(simple_lasso.predict(test[columns])[0] - legacy_lasso.predict(test[columns])[0]))
            ols_diff = float(abs(simple_ols.predict(test[columns])[0] - legacy_ols.predict(test[columns].fillna(medians))[0]))
            np.testing.assert_allclose(simple_lasso.predict(test[columns]), legacy_lasso.predict(test[columns]), rtol=0, atol=2e-9)
            np.testing.assert_allclose(simple_ols.predict(test[columns]), legacy_ols.predict(test[columns].fillna(medians)), rtol=0, atol=2e-9)
            np.testing.assert_allclose(simple_lasso.alpha, legacy_lasso.alpha_, rtol=1e-12, atol=0)
            checks.append(dict(origin=date, lasso_prediction_difference=lasso_diff,
                               ols_prediction_difference=ols_diff, alpha=simple_lasso.alpha))
    return checks


def compare(predictions, output):
    old = pd.read_csv(FROZEN / "predictions.csv")
    old = old.loc[old.experiment.str.startswith("expanded")].copy()
    for frame in [old, predictions]:
        for col in ["date", "target_date"]:
            frame[col] = pd.to_datetime(frame[col])
    # CSV round trips can change a refitted AR prediction at machine precision.
    # Verify numerical equivalence before aligning benchmark metadata for the
    # original strict pairing function. Forecast predictions are never altered.
    reference = old.loc[(old.pool == "baseline") & (old.model == "ar")].set_index(["experiment", "date"])
    if reference.index.has_duplicates:
        raise ValueError("Original baseline reference contains duplicate origins.")
    benchmark_max_difference = 0.0
    for i, row in predictions.iterrows():
        ref = reference.loc[(row.experiment, row.date)]
        if row.target_date != ref.target_date or row.actual != ref.actual:
            raise ValueError("Combined and original forecast targets differ.")
        difference = abs(row.benchmark - ref.benchmark)
        benchmark_max_difference = max(benchmark_max_difference, difference)
        np.testing.assert_allclose(row.benchmark, ref.benchmark, rtol=0, atol=2e-12)
        predictions.loc[i, "benchmark"] = ref.benchmark
    all_predictions = pd.concat([old, predictions], ignore_index=True)
    summary, pairs, sensitivity = [], [], []
    for period in ["main", "later"]:
        experiment = f"expanded_{period}"
        for j, (reference_pool, reference_model) in enumerate([
            ("combined", "ar"), ("baseline", "lasso"), ("macro", "lasso"), ("combined", "ols")]):
            matched = evaluate.pair(all_predictions, experiment, "combined", reference_pool, reference_model)
            ref, cand = matched.reference_loss.to_numpy(), matched.candidate_loss.to_numpy()
            label = f"combined_lasso_vs_{reference_pool}_{reference_model}_{period}"
            seed = 20261014 + 100 * (period == "later") + j
            interval, _, _, _ = evaluate.bootstrap(ref, cand, 6, 20000, seed)
            meta = dict(comparison_id=label, period=period, reference_pool=reference_pool,
                        reference_model=reference_model, n=len(matched),
                        reference_MSE=float(ref.mean()), combined_MSE=float(cand.mean()),
                        reduction=float(1-cand.mean()/ref.mean()), seed=seed)
            summary.append({**meta, **evaluate.hac(ref, cand), **interval})
            pairs.append(matched.assign(comparison_id=label))
            for block in [3, 12]:
                stats, _, _, _ = evaluate.bootstrap(ref, cand, block, 20000, seed + block)
                sensitivity.append({**meta, "block_length": block, **stats})
    summary = pd.DataFrame(summary)
    # These new comparisons form separate families from the earlier macro-only tests.
    for period in ["main", "later"]:
        mask = summary.period == period
        summary.loc[mask, "hac_p_holm_new_four_comparison_family"] = evaluate.holm(summary.loc[mask, "hac_p_two_sided"])
    summary.to_csv(output / "inference_summary.csv", index=False)
    pd.concat(pairs, ignore_index=True).to_csv(output / "paired_losses.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(output / "bootstrap_sensitivity.csv", index=False)
    old_metrics = pd.read_csv(FROZEN / "metrics.csv")
    old_metrics = old_metrics.loc[old_metrics.experiment.str.startswith("expanded")].copy()
    new_metrics = pd.read_csv(output / "metrics.csv")
    comparisons = pd.concat([old_metrics, new_metrics], ignore_index=True)
    comparisons["MSE_reduction_vs_AR_pct"] = comparisons.OOS_R2 * 100
    comparisons.to_csv(output / "model_comparison.csv", index=False)
    return summary, comparisons, benchmark_max_difference


def report(summary, comparisons, output, combined, checks):
    lines = ["# Combined baseline + macro forecasting run", "",
             "The combined pool contains all nine monthly baseline columns and all 99 eligible FRED-MD macro columns: 108 additional candidate predictors plus six unpenalized volatility controls and an intercept. All additional predictors are penalized equally in Lasso.", "",
             "This is a nested predictor expansion of the existing nine-variable baseline. Related economic series and alternative transformations are intentionally retained; columns are not removed because they are correlated.", "",
             "The frozen September 2026 inputs, predictor-specific delays, 240-month rolling windows, five chronological validation folds and 200-value penalty grid are unchanged. Medians, projections and scaling are fitted only on each training fold. The main sample has 240 origins (January 2001–December 2020); the later sample has 35 origins (January 2021–November 2023). The target is next-month annualized realized volatility in levels.", "",
             "## Forecast accuracy", "",
             "Positive MSE reduction means improvement over the matching AR forecast. MSE is in squared level-volatility units.", "",
             "| Period | Pool | Model | MSE | RMSE | MSE reduction vs AR |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for period in ["main", "later"]:
        for pool in ["baseline", "macro", "combined"]:
            for model in ["ols", "lasso"]:
                row = comparisons.loc[(comparisons.experiment == f"expanded_{period}") & (comparisons.pool == pool) & (comparisons.model == model)].iloc[0]
                lines.append(f"| {period.title()} | {pool} | {model.upper()} | {row.MSE:.9f} | {row.RMSE:.9f} | {100*row.OOS_R2:+.3f}% |")
    lines += ["", "## Combined Lasso: matched comparisons", "",
              "| Period | Reference | MSE reduction | 95% paired block bootstrap interval | HAC p | Holm-adjusted p |",
              "| --- | --- | ---: | --- | ---: | ---: |"]
    for row in summary.itertuples():
        reference = "AR" if row.reference_model == "ar" else f"{row.reference_pool} {row.reference_model.upper()}"
        p = f"{row.hac_p_two_sided:.3f}" if np.isfinite(row.hac_p_two_sided) else "—"
        adjusted = f"{row.hac_p_holm_new_four_comparison_family:.3f}" if np.isfinite(row.hac_p_holm_new_four_comparison_family) else "—"
        lines.append(f"| {row.period.title()} | {reference} | {100*row.reduction:+.3f}% | [{100*row.bootstrap_reduction_ci_low:+.3f}%, {100*row.bootstrap_reduction_ci_high:+.3f}%] | {p} | {adjusted} |")
    lines += ["", "Inference is exploratory and retrospective. Intervals use 20,000 paired circular-block resamples with six-month blocks, with three- and twelve-month sensitivity results saved separately. HAC tests are two-sided tests of mean loss differences. Holm adjustment is applied separately to the four new comparisons in each period; it does not revise the thesis's existing test families. The later sample is not an untouched confirmation sample. These intervals resample saved forecast losses and do not refit or retune models. The later combined-versus-macro forecasts are numerically identical, so a loss-difference test is not reported for that comparison.", "", "## Predictor selection", ""]
    coefficients = pd.read_csv(output / "coefficients.csv")
    selection = pd.read_csv(output / "selection.csv")
    for period in ["main", "later"]:
        rows = coefficients.loc[(coefficients.experiment == f"expanded_{period}") & (coefficients.model == "lasso")]
        counts = (rows[combined].abs() > 1e-12).sum(axis=1)
        lines.append(f"{period.title()}: zero extra predictors in {int((counts == 0).sum())}/{len(counts)} windows; mean selected extras {counts.mean():.3f}; maximum {counts.max()}.")
        lines.append("")
        top = selection.loc[(selection.experiment == f"expanded_{period}") & (selection.selected_n > 0)].sort_values("selected_n", ascending=False)
        for row in top.head(10).itertuples():
            lines.append(f"- {row.feature}: {row.selected_n}/{len(rows)} windows ({100*row.selected_share:.2f}%).")
        lines.append("")
    lines += ["## Validation and reproducibility", "",
              "- Frozen inputs and original method files match the recorded thesis hashes; scientific package versions match the existing run.",
              "- Independent original-estimator checks agree for both OLS and Lasso at the first main and first later origins (absolute prediction tolerance 2e-9); selected Lasso penalties agree.",
              "- All 275 origins are retained; targets and comparison dates match the existing baseline and macro runs. Recomputed AR benchmarks agree within 2e-12; machine-precision CSV round-trip differences are aligned only in comparison metadata, without changing forecast predictions. All predictions are finite and coefficient exports reconstruct the saved forecasts.",
              "- Every original input and method file is checked again after the run to confirm unchanged contents.", "",
              "Run `run_combined.py` with the original project's scientific Python environment. `run_manifest.json` records source hashes, software, method settings and artifact hashes. Forecasts, coefficients, window diagnostics, selections, accuracy comparisons and loss inference are saved alongside this report."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    global FROZEN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=FROZEN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Separate smoke run: first N origins per period.")
    parser.add_argument("--reuse", action="store_true", help="Recheck and compare completed forecasts without refitting.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    FROZEN = args.input_dir.resolve()
    output = args.output.resolve()
    if output == FROZEN or output.is_relative_to(PACKAGE / "results" / "reference"):
        parser.error("Output must be separate from the submitted reference results.")
    output.mkdir(parents=True, exist_ok=True)
    base, expanded, pools, combined, before = load_inputs()
    (output / "pools.json").write_text(json.dumps({**pools, "combined": combined}, indent=2) + "\n")
    checks = validate_estimators(expanded, combined)
    print("Independent estimator checks passed:", checks, flush=True)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        if args.reuse:
            predictions = pd.read_csv(output / "predictions.csv", parse_dates=["date", "target_date"])
        else:
            predictions = forecast.run(base, expanded, {"combined": combined}, output,
                                       limit=args.limit, selected="expanded")
    reconstruction_max = 0.0
    coefficients = pd.read_csv(output / "coefficients.csv")
    windows = pd.read_csv(output / "windows.csv")
    for period in ["main", "later"]:
        rows = predictions.loc[predictions.experiment == f"expanded_{period}"]
        expected_n = min(args.limit, 240 if period == "main" else 35) if args.limit else (240 if period == "main" else 35)
        assert len(rows) == expected_n * 3 and not rows.duplicated(["date", "model"]).any()
        for row in rows.itertuples():
            i = expanded.index.get_loc(row.date)
            train, test = expanded.iloc[i-240:i], expanded.iloc[i]
            columns = settings.AR if row.model == "ar" else settings.AR + combined
            slopes = coefficients.loc[(coefficients.experiment == row.experiment) & (coefficients.date == str(row.date.date())) & (coefficients.model == row.model)].iloc[0]
            values = test[columns].fillna(train[columns].median().fillna(0))
            reconstructed = float(slopes.intercept + np.dot(values, slopes[columns]))
            reconstruction_max = max(reconstruction_max, abs(reconstructed - row.prediction))
        assert (rows.pool_size == 108).all()
    assert reconstruction_max <= 2e-9
    assert (windows.training_n == 240).all()
    assert (pd.to_datetime(windows.training_end) < pd.to_datetime(windows.date)).all()
    assert windows.lasso_relative_gap.max() <= 1.1e-6
    summary = None
    benchmark_difference = None
    if not args.limit:
        summary, comparisons, benchmark_difference = compare(predictions, output)
        report(summary, comparisons, output, combined, checks)
    if any(digest(Path(path)) != value for path, value in before.items()):
        raise RuntimeError("An original input or method file changed during the run.")
    manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(), complete=args.limit is None,
        additional_predictors=108, unpenalized_AR_controls=6, intercept=True, rolling_window=240,
        cv_folds=5, alphas=settings.ALPHAS.tolist(), solver="lars", lasso_tolerance=1e-6,
        periods=settings.PERIODS, bootstrap_resamples=20000, bootstrap_block_length=6,
        inference_status="exploratory retrospective additional run",
        input_and_method_sha256=before, runner_sha256=digest(Path(__file__)),
        software={name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
        reused_forecasts=args.reuse,
        validation=dict(estimator_checks=checks, forecast_rows=len(predictions),
            coefficient_reconstruction_max_difference=reconstruction_max,
            max_selected_lasso_relative_gap=float(windows.lasso_relative_gap.max()),
            maximum_difference_from_existing_AR_benchmark=benchmark_difference,
            original_inputs_unchanged=True),
        warnings=None if args.reuse else [str(w.message) for w in recorded],
        warning_capture="not recaptured when reusing saved forecasts" if args.reuse else "forecast fitting")
    manifest["artifact_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "run_manifest.json"}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    if summary is not None:
        print(summary[["period", "reference_pool", "reference_model", "reduction", "hac_p_two_sided"]].to_string(index=False), flush=True)
    print(f"Saved combined-pool outputs to {output}", flush=True)


if __name__ == "__main__":
    main()
