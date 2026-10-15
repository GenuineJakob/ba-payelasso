"""Uncertainty for selected saved monthly forecasts; never refits a model.

Run from Code: python src/forecast_inference.py
Positive differences/reductions favor Lasso. Primary comparisons are selected
after inspection, not prespecified. GW refers to unconditional section 3.4,
not a conditional predictive ability test or a test of individual coefficients.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20261013
DEFAULT_RESAMPLES = 20000
BLOCK_LENGTHS = (3, 6, 12)
GW_SOURCE = "https://economia.uc3m.es/jgonzalo/teaching/PhdTimeSeries/GiacominiWhite.pdf"
BOOTSTRAP_SOURCE = "https://mathweb.ucsd.edu/~politis/impactBOOT.pdf"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loss_arrays(reference, candidate):
    """Accept already aligned squared errors, never independently resample them."""
    reference, candidate = np.asarray(reference, float), np.asarray(candidate, float)
    if reference.ndim != 1 or reference.shape != candidate.shape or len(reference) < 2:
        raise ValueError("Paired one-dimensional losses need at least two observations.")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Nonfinite losses are not eligible for inference.")
    if np.any(reference < 0) or np.any(candidate < 0) or reference.mean() <= 0:
        raise ValueError("Losses must be nonnegative with a positive reference MSE.")
    return reference, candidate


def hac_mean_test(reference, candidate, lags: int | None = None, confidence=0.95):
    """Two-sided unconditional mean-loss test with centered Bartlett HAC.

    d_t = reference_loss - candidate_loss; z = mean(d)/sqrt(LRV/n).
    LRV = gamma_0 + 2 sum_(j=1)^q (1-j/(q+1))*gamma_j;
    gamma_j uses centered d and denominator n (no degrees-of-freedom correction).
    Default q=floor(4*(n/100)^(2/9)), a deterministic bandwidth heuristic;
    explicit fixed lags are finite-sample sensitivities. No horizon-1 truncation.
    Degenerate variance yields unavailable inference, never a fabricated p=1.
    """
    reference, candidate = loss_arrays(reference, candidate)
    n = len(reference)
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one.")
    if lags is None:
        lags = min(n - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    if not isinstance(lags, (int, np.integer)) or not 0 <= lags < n:
        raise ValueError("HAC lags must be an integer between zero and n-1.")
    differential = reference - candidate
    mean = float(differential.mean())
    centered = differential - mean
    lrv = float(centered @ centered / n)
    for lag in range(1, lags + 1):
        lrv += 2 * (1 - lag / (lags + 1)) * float(centered[lag:] @ centered[:-lag] / n)
    # A relative threshold is invariant to the response's units. Identical or
    # effectively constant differences violate the positive-variance assumption.
    scale = max(float(np.mean(differential**2)), np.finfo(float).tiny)
    status = "ok"
    if np.max(np.abs(differential)) <= 1e-12 * max(reference.mean(), candidate.mean()):
        status = "numerically_identical_losses"
    elif lrv <= 1e-14 * scale:
        status = "degenerate_loss_variance"
    se = np.sqrt(max(lrv, 0) / n) if status == "ok" else np.nan
    z = mean / se if status == "ok" else np.nan
    critical = float(norm.ppf((1 + confidence) / 2))
    return {"hac_lags": int(lags), "mean_loss_difference": mean,
            "hac_long_run_variance": lrv, "hac_standard_error": float(se),
            "hac_z": float(z), "hac_p_two_sided": float(2 * norm.sf(abs(z))),
            "hac_difference_ci_low": float(mean - critical * se),
            "hac_difference_ci_high": float(mean + critical * se), "test_status": status}


def circular_block_indices(n: int, block_length: int, resamples: int, rng):
    """Uniform circular starts; concatenate consecutive blocks and trim to n."""
    if not all(isinstance(v, (int, np.integer)) for v in (n, block_length, resamples)):
        raise ValueError("Bootstrap sizes must be integers.")
    if n < 2 or not 1 <= block_length <= n or resamples < 1:
        raise ValueError("Invalid bootstrap sizes.")
    starts = rng.integers(0, n, size=(resamples, (n + block_length - 1) // block_length))
    indices = (starts[:, :, None] + np.arange(block_length)) % n
    return indices.reshape(resamples, -1)[:, :n]


def paired_block_interval(reference, candidate, block_length=6, resamples=DEFAULT_RESAMPLES,
                          seed=DEFAULT_SEED, confidence=0.95):
    """Percentile CIs for MSE difference and 1-MSE_candidate/MSE_reference.

    Same sampled dates for both series; the denominator is recomputed in each
    replicate. Sampling saved losses does not redo fitting/tuning/model search.
    Weak dependence and approximate stationarity of the joint losses are needed.
    """
    reference, candidate = loss_arrays(reference, candidate)
    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence level.")
    rng = np.random.default_rng(seed)
    indices = circular_block_indices(len(reference), block_length, resamples, rng)
    ref_means = reference[indices].mean(axis=1)
    candidate_means = candidate[indices].mean(axis=1)
    # Do not silently drop pathological replicates and report a conditional CI.
    if np.any(ref_means <= 0):
        return {"bootstrap_status": "zero_reference_mse_in_resamples",
                **{key: np.nan for key in ["bootstrap_reduction_ci_low", "bootstrap_reduction_ci_high",
                                           "bootstrap_difference_ci_low", "bootstrap_difference_ci_high"]}}
    quantiles = [(1 - confidence) / 2, (1 + confidence) / 2]
    reduction_ci = np.quantile(1 - candidate_means / ref_means, quantiles)
    difference_ci = np.quantile(ref_means - candidate_means, quantiles)
    return {"bootstrap_status": "ok", "bootstrap_reduction_ci_low": float(reduction_ci[0]),
            "bootstrap_reduction_ci_high": float(reduction_ci[1]),
            "bootstrap_difference_ci_low": float(difference_ci[0]),
            "bootstrap_difference_ci_high": float(difference_ci[1])}


def holm_adjust(p_values):
    """Holm FWER adjustment; undefined tests remain undefined and count in family."""
    p = np.asarray(p_values, float)
    finite = np.isfinite(p)
    if np.any((p[finite] < 0) | (p[finite] > 1)):
        raise ValueError("p-values must be in [0,1].")
    result = np.full(p.shape, np.nan)
    order = np.flatnonzero(finite)[np.argsort(p[finite], kind="stable")]
    if len(order):
        result[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(order))) * p[order]))
    return result


def validate_monthly_pairs(frame):
    frame = frame.copy()
    required = ["date", "target_date", "actual", "candidate", "reference"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Missing paired-forecast columns.")
    for column in ["date", "target_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="raise")
        if frame[column].isna().any():
            raise ValueError("Missing forecast dates.")
    frame = frame.sort_values("date").reset_index(drop=True)
    if len(frame) < 2 or frame.date.duplicated().any() or frame.target_date.duplicated().any():
        raise ValueError("Missing or duplicate paired observations.")
    expected = pd.date_range(frame.date.iloc[0], frame.date.iloc[-1], freq="ME")
    if not pd.DatetimeIndex(frame.date).equals(expected):
        raise ValueError("Forecast origins must be consecutive calendar month ends.")
    if not (frame.target_date == frame.date + pd.offsets.MonthEnd(1)).all():
        raise ValueError("Target dates must be one month after forecast origins.")
    if not np.isfinite(frame[["actual", "candidate", "reference"]].to_numpy(float)).all():
        raise ValueError("Nonfinite paired forecasts.")
    frame["reference_loss"] = (frame.actual - frame.reference)**2
    frame["candidate_loss"] = (frame.actual - frame.candidate)**2
    frame["loss_difference"] = frame.reference_loss - frame.candidate_loss
    loss_arrays(frame.reference_loss, frame.candidate_loss)
    return frame


def load_comparisons(root, sources):
    def read(relative, kind="csv"):
        path = root / relative
        sources[relative] = sha256(path)
        return pd.read_csv(path) if kind == "csv" else json.loads(path.read_text())

    comparisons = []
    for design in ("rolling", "fixed"):
        predictions = read(f"output/results/{design}_predictions.csv")
        # The legacy fixed export stores an unnamed DatetimeIndex column.
        if "date" not in predictions and str(predictions.columns[0]).startswith("Unnamed:"):
            predictions = predictions.rename(columns={predictions.columns[0]: "date"})
        table = read(f"output/tables/{design}_model_comparison.csv")
        predictions["target_date"] = pd.to_datetime(predictions.date) + pd.offsets.MonthEnd(1)
        pairs = validate_monthly_pairs(predictions.rename(columns={"lasso": "candidate", "benchmark": "reference"}))
        np.testing.assert_allclose(pairs.candidate_loss.mean(), table.loc[table.model == "lasso", "MSE"].item(), rtol=1e-9)
        np.testing.assert_allclose(pairs.reference_loss.mean(), table.loc[table.model == "benchmark", "MSE"].item(), rtol=1e-9)
        comparisons.append(({
            "comparison_id": f"original_{design}_lasso_vs_ar", "period": "main",
            "role": "primary" if design == "rolling" else "diagnostic",
            "comparison": f"Original {design} Lasso vs AR(6)",
            "candidate_model": "lasso", "reference_model": "ar6_benchmark", "pool": "original_ten",
            "window": "360 months" if design == "rolling" else "Fixed fit through 2000",
            "source": f"output/results/{design}_predictions.csv",
        }, pairs))

    for period in ("main", "holdout"):
        predictions = read(f"output/results/fredmd_predictions_{period}.csv")
        table = read(f"output/tables/fredmd_model_comparison_{period}.csv")
        manifest = read(f"output/results/fredmd_run_manifest_{period}.json", "json")
        windows = read(f"output/tables/fredmd_window_diagnostics_{period}.csv")
        if manifest["smoke_run"] or manifest["window_size"] != 240 or manifest["n_splits"] != 5:
            raise ValueError("Inference requires full, five-fold 240-month FRED-MD evaluations.")
        if not (windows.loc[windows.pool == "fredmd_macro", "training_n"] == 240).all():
            raise ValueError("Training window differs from the stated fixed-length design.")
        pool = predictions.loc[predictions.pool == "fredmd_macro"].copy()
        if pool.empty or set(pool.period) != {period} or pool.duplicated(["date", "model"]).any():
            raise ValueError("Invalid macro-only prediction pool.")
        candidate = pool.loc[pool.model == "lasso"].set_index("date").sort_index()
        if len(candidate) != manifest["n_forecasts"]:
            raise ValueError("Forecast count differs from its run manifest.")
        for reference_model in ("ar6_benchmark", "ols"):
            reference = pool.loc[pool.model == reference_model].set_index("date").sort_index()
            if not candidate.index.equals(reference.index):
                raise ValueError("Models must have identical dates; no inner-join sample loss.")
            if not candidate.target_date.equals(reference.target_date):
                raise ValueError("Target dates differ across paired models.")
            np.testing.assert_allclose(candidate.actual, reference.actual, rtol=0, atol=0)
            np.testing.assert_allclose(candidate.benchmark, reference.benchmark, rtol=0, atol=0)
            if reference_model == "ar6_benchmark":
                np.testing.assert_allclose(reference.prediction, reference.benchmark, rtol=0, atol=0)
            paired = candidate[["target_date", "actual", "prediction"]].rename(columns={"prediction": "candidate"})
            paired["reference"] = reference.prediction
            paired = validate_monthly_pairs(paired.reset_index())
            for model, losses in [("lasso", paired.candidate_loss), (reference_model, paired.reference_loss)]:
                mse = table.loc[(table.pool == "fredmd_macro") & (table.model == model), "MSE"].item()
                np.testing.assert_allclose(losses.mean(), mse, rtol=1e-9, atol=1e-14)
            comparisons.append(({
                "comparison_id": f"macro_lasso_vs_{'ar' if reference_model == 'ar6_benchmark' else 'ols'}_{period}",
                "comparison": f"Expanded macro Lasso vs {'AR(6)' if reference_model == 'ar6_benchmark' else 'OLS'}",
                "period": "main" if period == "main" else "later",
                "role": "primary" if period == "main" else "retrospective_robustness",
                "candidate_model": "lasso", "reference_model": reference_model, "pool": "fredmd_macro",
                "window": "240 months", "source": f"output/results/fredmd_predictions_{period}.csv",
            }, paired))
    return comparisons


def interpretation(row):
    if row.test_status != "ok":
        return "No valid test: " + row.test_status.replace("_", " ")
    if row.hac_p_holm < 0.05:
        return "Lower Lasso MSE (Holm 5%)" if row.mean_loss_difference > 0 else "Higher Lasso MSE (Holm 5%)"
    return "Improvement not established (Holm 5%)"


def display_table(frame):
    rows = []
    for _, row in frame.iterrows():
        p = lambda value: "Unavailable" if pd.isna(value) else f"{value:.4f}"
        interval = ("Unavailable" if pd.isna(row.bootstrap_reduction_ci_low) else
                    f"[{row.bootstrap_reduction_ci_low*100:+.3f}%, {row.bootstrap_reduction_ci_high*100:+.3f}%]")
        rows.append([row.comparison, row.period, int(row.n), f"{row.reduction*100:+.3f}%", interval,
                     p(row.hac_p_two_sided), p(row.hac_p_holm), interpretation(row)])
    return pd.DataFrame(rows, columns=["Comparison", "Period", "N", "MSE reduction vs reference",
                                      "95% block-bootstrap CI", "HAC p (two-sided)", "Holm p", "Assessment"])


def markdown_table(frame):
    # Avoid a new dependency on tabulate.
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"]*len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))


def methodology_paragraphs(manifest):
    config = manifest["configuration"]
    return [
        "Three primary comparisons are selected from already inspected results: original 360-month rolling Lasso vs AR(6), expanded 99-variable macro-only Lasso vs AR(6), and that same expanded Lasso vs corrected OLS. Each pair uses its own identical origin dates, target dates and actuals. Original and expanded designs have different windows and samples; they are not pooled or compared as a common-sample experiment. Fixed Lasso is a diagnostic; the two later-period expanded comparisons are retrospective robustness.",
        "Positive d_t = reference squared error − Lasso squared error favors Lasso. MSE reduction = 1 − mean(Lasso loss)/mean(reference loss). The OLS comparison uses OLS, not AR, in the denominator. Forecast targets are annualized volatility in levels, one month ahead. Percentages describe error reductions, not explained variance.",
        "The test is the two-sided unconditional mean-loss comparison in Giacomini–White (2006), section 3.4, under a fixed estimation-window framework. It uses z = mean(d)/sqrt(HAC long-run variance/n), a standard-normal reference, centered autocovariances with denominator n and Bartlett weights. The default bandwidth is floor(4*(n/100)^(2/9)); lags 0, 3, 6 and 12 are reported as finite-sample sensitivities. This is not the conditional GW instrument test and does not assume loss differences are independent merely because forecasts are one-step ahead. [Method and assumptions](" + GW_SOURCE + ").",
        "Fixed-length rolling fitting and fold-local tuning form the forecasting procedure being compared; the GW framework permits general estimation methods. Validity still requires weak dependence, sufficient moments and nonzero long-run variance. Heavy crisis losses, short evaluation samples and departures from these assumptions can affect the approximation. A nearly identical or constant loss differential produces an unavailable test, not an arbitrary p-value. No Clark–West adjustment or Lasso coefficient significance test is applied.",
        f"Paired circular block bootstrap: {config['resamples']:,} resamples, random seed {config['seed']}, {config['confidence']*100:.0f}% percentile intervals, main block length {config['primary_block_length']} months; lengths 3, 6 and 12 are shown as sensitivity. Uniform random block starts wrap at the sample end; consecutive blocks are concatenated and trimmed to n. Both losses use the same sampled dates, and each replicate recomputes both MSEs and their ratio. This assumes approximate stationarity and weak dependence of the joint saved losses. [Block-bootstrap background](" + BOOTSTRAP_SOURCE + ").",
        "The bootstrap resamples saved losses; it does not refit models, retune alpha or repeat the historical model search. The intervals do not capture all uncertainty from model selection and are pointwise, not simultaneous. They are not intervals for future individual volatility observations. The HAC and bootstrap procedures use different approximations, so their decisions need not agree exactly.",
        "Holm adjustment controls multiplicity across the three selected primary HAC tests. The two later-period tests have a separate exploratory family. Sensitivity results are not additional confirmatory tests. This adjustment does not correct the earlier search across models, grids or periods: these are selected, retrospective comparisons, not a prespecified confirmatory exercise. Later results have only 35 observations and are not an untouched holdout.",
        "A confidence interval including zero or a non-rejection does not establish equivalence or prove that macroeconomic information has zero forecasting value. A positive significant difference supports lower average squared error for the compared procedures under the stated assumptions; it identifies neither causation nor significant individual macroeconomic coefficients.",
    ]


def build_inference_document(primary, robustness, diagnostic, manifest):
    blocks = ["# Forecast uncertainty for the selected thesis comparisons",
              "Generated " + manifest["generated_utc"] + ". Existing forecast files are read without refitting or modification.",
              "## Primary comparisons", markdown_table(display_table(primary)),
              "## Later-period robustness", markdown_table(display_table(robustness)),
              "## Fixed-model diagnostic", markdown_table(display_table(diagnostic)),
              "## Method, scope and limitations"]
    blocks.extend(methodology_paragraphs(manifest))
    blocks += ["## Sensitivity and reproducibility",
               "The full block-length and HAC-lag sensitivity tables are in [forecast_inference_bootstrap_sensitivity.csv](../tables/forecast_inference_bootstrap_sensitivity.csv) and [forecast_inference_hac_sensitivity.csv](../tables/forecast_inference_hac_sensitivity.csv). Paired losses and dates are exported in [forecast_inference_losses.csv](../results/forecast_inference_losses.csv). All configuration, source checksums and output checksums are in [forecast_inference_manifest.json](../tables/forecast_inference_manifest.json).",
               "Run from Code: `python src/forecast_inference.py`, then `python src/build_project_report.py`. The default seed and configuration reproduce the CSV results; timestamps can change. No training is performed."]
    return "\n\n".join(blocks) + "\n"


def run(root=ROOT, resamples=DEFAULT_RESAMPLES, seed=DEFAULT_SEED):
    if not isinstance(resamples, int) or resamples < 1000:
        raise ValueError("Use at least 1,000 bootstrap resamples.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    sources = {}
    comparisons = load_comparisons(root, sources)
    sources["src/forecast_inference.py"] = sha256(root / "src/forecast_inference.py")
    # Hash all existing prediction artifacts, not just the inference inputs.
    prediction_hashes = {str(p.relative_to(root)): sha256(p) for p in (root / "output/results").glob("*predictions*.csv")}
    rows, bootstrap_rows, hac_rows, loss_rows = [], [], [], []
    for index, (meta, pairs) in enumerate(comparisons):
        reference, candidate = pairs.reference_loss.to_numpy(), pairs.candidate_loss.to_numpy()
        base = {**meta, "n": len(pairs), "scale": "level", "first_origin": pairs.date.iloc[0].strftime("%Y-%m-%d"),
                "last_origin": pairs.date.iloc[-1].strftime("%Y-%m-%d"),
                "first_target": pairs.target_date.iloc[0].strftime("%Y-%m-%d"),
                "last_target": pairs.target_date.iloc[-1].strftime("%Y-%m-%d"),
                "reference_mse": float(reference.mean()), "candidate_mse": float(candidate.mean()),
                "reduction": float(1 - candidate.mean()/reference.mean())}
        hac = hac_mean_test(reference, candidate)
        boot = {}
        for block in BLOCK_LENGTHS:
            bootstrap_seed = seed + 100 * index + block
            interval = paired_block_interval(reference, candidate, block, resamples, bootstrap_seed)
            bootstrap_rows.append({**base, "block_length": block, "resamples": resamples,
                                   "seed": bootstrap_seed, **interval})
            if block == 6:
                boot = interval
        rows.append({**base, **hac, "block_length": 6, "resamples": resamples,
                     "seed": seed + 100*index + 6, **boot})
        for lag in sorted(set([0, 3, 6, 12, hac["hac_lags"]])):
            hac_rows.append({**base, "default_bandwidth": lag == hac["hac_lags"],
                             **hac_mean_test(reference, candidate, lag)})
        loss_rows.append(pairs.assign(comparison_id=meta["comparison_id"], period=meta["period"], role=meta["role"]))
    summary = pd.DataFrame(rows)
    summary["hac_p_holm"] = np.nan
    for role in ("primary", "retrospective_robustness"):
        selected = summary.role == role
        summary.loc[selected, "hac_p_holm"] = holm_adjust(summary.loc[selected, "hac_p_two_sided"])
    summary["assessment"] = summary.apply(interpretation, axis=1)
    paths = {}
    output_frames = {
        "output/tables/forecast_inference_summary.csv": summary,
        "output/tables/forecast_inference_bootstrap_sensitivity.csv": pd.DataFrame(bootstrap_rows),
        "output/tables/forecast_inference_hac_sensitivity.csv": pd.DataFrame(hac_rows),
        "output/results/forecast_inference_losses.csv": pd.concat(loss_rows, ignore_index=True),
    }
    for relative, frame in output_frames.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, float_format="%.17g")
        paths[relative] = sha256(path)
    for relative, digest in {**sources, **prediction_hashes}.items():
        if sha256(root / relative) != digest:
            raise RuntimeError(f"Input changed during inference: {relative}")
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configuration": {"resamples": resamples, "seed": seed, "confidence": 0.95,
                          "primary_block_length": 6, "sensitivity_block_lengths": list(BLOCK_LENGTHS),
                          "hac_bandwidth": "floor(4*(n/100)^(2/9))", "hac_sensitivity_lags": [0, 3, 6, 12],
                          "bootstrap_method": "paired circular block, percentile",
                          "hac_method": "centered Bartlett autocovariances, denominator n, no small-sample correction",
                          "test": "two-sided unconditional Giacomini-White section 3.4 mean-loss test",
                          "holm_families": {"primary": 3, "retrospective_robustness": 2}},
        "selection_status": "comparisons selected after inspecting results; not prespecified",
        "bootstrap_refits_models": False, "training_rerun": False, "causal_identification": False,
        "sources_sha256": sources, "unchanged_prediction_files_sha256": prediction_hashes,
        "outputs_sha256": paths,
        "software_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                              "scipy": __import__("scipy").__version__},
        "references": {"giacomini_white_2006_section_3_4": GW_SOURCE,
                       "block_bootstrap_author_background": BOOTSTRAP_SOURCE},
    }
    document = build_inference_document(summary.loc[summary.role == "primary"],
                                        summary.loc[summary.role == "retrospective_robustness"],
                                        summary.loc[summary.role == "diagnostic"], manifest)
    report = root / "output/reports/forecast_inference.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(document)
    manifest["outputs_sha256"][str(report.relative_to(root))] = sha256(report)
    (root / "output/tables/forecast_inference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(display_table(summary).to_string(index=False))
    print(f"Verified {len(prediction_hashes)} existing prediction files unchanged.")
    return summary, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run(args.root, args.resamples, args.seed)


if __name__ == "__main__":
    main()
