"""Matched forecast losses, HAC tests, paired bootstrap intervals and figures.

The four main and three later Holm families are retrospective, as in the thesis.
The fixed comparison is diagnostic. Bootstrap draws never refit or retune models.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, skew

from settings import RESAMPLES, SEED


def pair(predictions, experiment, candidate_pool, reference_pool, reference_model):
    data = predictions.loc[predictions.experiment == experiment]
    candidate = data.loc[(data.pool == candidate_pool) & (data.model == "lasso")].set_index("date").sort_index()
    reference = data.loc[(data.pool == reference_pool) & (data.model == reference_model)].set_index("date").sort_index()
    if candidate.empty or candidate.index.has_duplicates or reference.index.has_duplicates or not candidate.index.equals(reference.index):
        raise ValueError("Compare all identical dates; never discard unmatched forecasts.")
    for column in ["actual", "benchmark", "target_date"]:
        if not candidate[column].equals(reference[column]):
            raise ValueError(f"Paired {column} values differ.")
    expected = pd.date_range(candidate.index[0], candidate.index[-1], freq="ME")
    if not candidate.index.equals(expected) or not (candidate.target_date == candidate.index + pd.offsets.MonthEnd(1)).all():
        raise ValueError("Origins must be consecutive and targets one month later.")
    result = candidate[["target_date", "actual", "prediction"]].rename(columns={"prediction": "candidate"})
    result["reference"] = reference.prediction
    result["reference_loss"] = (result.actual - result.reference)**2
    result["candidate_loss"] = (result.actual - result.candidate)**2
    result["loss_difference"] = result.reference_loss - result.candidate_loss
    if not np.isfinite(result.select_dtypes("number")).all().all() or result.reference_loss.mean() <= 0:
        raise ValueError("Inference requires finite paired losses and positive reference MSE.")
    return result.reset_index()


def hac(reference, candidate, lags=None):
    d = np.asarray(reference) - np.asarray(candidate)
    n = len(d)
    if n < 2:
        raise ValueError("Inference needs at least two matched observations.")
    q = min(n - 1, int(np.floor(4 * (n / 100)**(2 / 9)))) if lags is None else lags
    if not 0 <= q < n:
        raise ValueError("HAC lag must be between zero and n-1.")
    centered = d - d.mean()
    variance = float(centered @ centered / n)
    for j in range(1, q + 1):
        variance += 2 * (1 - j / (q + 1)) * float(centered[j:] @ centered[:-j] / n)
    status = "ok"
    if np.max(np.abs(d)) <= 1e-12 * max(np.mean(reference), np.mean(candidate)):
        status = "numerically_identical_losses"
    elif variance <= 1e-14 * max(float(np.mean(d**2)), np.finfo(float).tiny):
        status = "degenerate_loss_variance"
    se = np.sqrt(max(variance, 0) / n) if status == "ok" else np.nan
    z = float(d.mean() / se) if status == "ok" else np.nan
    return dict(hac_lags=q, mean_loss_difference=float(d.mean()), hac_standard_error=float(se),
                hac_z=z, hac_p_two_sided=float(2 * norm.sf(abs(z))), test_status=status,
                hac_difference_ci_low=float(d.mean() - norm.ppf(.975)*se),
                hac_difference_ci_high=float(d.mean() + norm.ppf(.975)*se))


def bootstrap(reference, candidate, block, resamples, seed):
    n = len(reference)
    if not 1 <= block <= n or resamples < 1:
        raise ValueError("Invalid block length or bootstrap count.")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(resamples, (n + block - 1) // block))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(resamples, -1)[:, :n]
    ref = np.asarray(reference)[indices].mean(axis=1)
    cand = np.asarray(candidate)[indices].mean(axis=1)
    if (ref <= 0).any():
        raise ValueError("A bootstrap reference MSE is zero; do not discard that draw.")
    reduction, difference = 1 - cand/ref, ref-cand
    rhat, dhat = 1-np.mean(candidate)/np.mean(reference), np.mean(reference)-np.mean(candidate)
    rlo, rhi = np.quantile(reduction, [.025, .975])
    dlo, dhi = np.quantile(difference, [.025, .975])
    summary = dict(bootstrap_reduction_ci_low=rlo, bootstrap_reduction_ci_high=rhi,
        bootstrap_difference_ci_low=dlo, bootstrap_difference_ci_high=dhi,
        basic_reduction_ci_low=2*rhat-rhi, basic_reduction_ci_high=2*rhat-rlo,
        basic_difference_ci_low=2*dhat-dhi, basic_difference_ci_high=2*dhat-dlo)
    return summary, indices, reduction, difference


def holm(values):
    p = np.asarray(values, dtype=float)
    result = np.full(p.shape, np.nan)
    order = np.flatnonzero(np.isfinite(p))
    order = order[np.argsort(p[order], kind="stable")]
    if len(order):
        result[order] = np.minimum(1, np.maximum.accumulate((len(p)-np.arange(len(order)))*p[order]))
    return result


def run(predictions, base, output, resamples=RESAMPLES):
    predictions = predictions.copy()
    for column in ["date", "target_date"]:
        predictions[column] = pd.to_datetime(predictions[column])
    comparisons = [("original_rolling_lasso_vs_ar", "original_rolling", "original", "original", "ar", "primary"),
                   ("original_fixed_lasso_vs_ar", "original_fixed", "original", "original", "ar", "diagnostic")]
    for period in ["main", "later"]:
        suffix = "main" if period == "main" else "holdout"
        role = "primary" if period == "main" else "retrospective_robustness"
        for model in ["ar", "ols"]:
            comparisons.append((f"macro_lasso_vs_{model}_{suffix}", f"expanded_{period}", "macro", "macro", model, role))
    for period in ["main", "later"]:
        suffix = "main" if period == "main" else "holdout"
        comparisons.append((f"macro_lasso_vs_baseline_{suffix}", f"expanded_{period}", "macro", "baseline", "lasso",
                            "primary" if period == "main" else "retrospective_robustness"))
    summaries, sensitivity, hac_sensitivity, paired, crisis_groups = [], [], [], [], []
    main_pairs, main_draws, distribution = None, None, None
    for i, (label, experiment, pool, ref_pool, model, role) in enumerate(comparisons):
        pairs = pair(predictions, experiment, pool, ref_pool, model)
        expected_n = 229 if experiment == "original_rolling" else (35 if experiment.endswith("later") else 240)
        if len(pairs) != expected_n:
            raise ValueError("Inference requires complete evaluations, never a smoke run.")
        ref, cand = pairs.reference_loss.to_numpy(), pairs.candidate_loss.to_numpy()
        meta = dict(comparison_id=label, experiment=experiment, role=role, n=len(pairs),
                    reference_mse=float(ref.mean()), candidate_mse=float(cand.mean()),
                    reduction=float(1-cand.mean()/ref.mean()))
        h = hac(ref, cand)
        for block in [3, 6, 12]:
            seed = SEED + 100*i + block
            interval, indices, reductions, differences = bootstrap(ref, cand, block, resamples, seed)
            sensitivity.append({**meta, "block_length": block, "resamples": resamples, "seed": seed, **interval})
            if block == 6:
                summaries.append({**meta, **h, "block_length": block, "resamples": resamples, "seed": seed, **interval})
            if label == "macro_lasso_vs_ar_main" and block == 6:
                main_pairs = pairs
                d = ref-cand
                tolerance = 1e-12*max(ref.mean(), cand.mean())
                events = pairs.target_date.isin(pd.to_datetime(["2020-04-30", "2020-05-31"])).to_numpy()
                count = events[indices].sum(axis=1)
                main_draws = pd.DataFrame(dict(reduction=reductions, mean_loss_difference=differences,
                    april_may_2020_draw_count=count, active_loss_draw_count=(np.abs(d)>tolerance)[indices].sum(axis=1)))
                distribution = dict(n=len(d), zero_count=int((np.abs(d)<=tolerance).sum()),
                    positive_count=int((d>tolerance).sum()), negative_count=int((d < -tolerance).sum()),
                    mean=float(d.mean()), median=float(np.median(d)), skewness=float(skew(d, bias=False)),
                    april_may_net_share=float(d[events].sum()/d.sum()) if d.sum() else None,
                    without_april_may_reduction=float(1-cand[~events].mean()/ref[~events].mean()))
                for group, mask in [("all", np.ones(resamples, dtype=bool)), ("no_April_May", count==0), ("with_April_May", count>0)]:
                    lo, hi = np.quantile(reductions[mask], [.025, .975]) if mask.any() else [np.nan, np.nan]
                    crisis_groups.append(dict(group=group, replicates=int(mask.sum()), fraction=float(mask.mean()),
                                              percentile_low=lo, percentile_high=hi,
                                              nonpositive_fraction=float((reductions[mask]<=0).mean()) if mask.any() else np.nan))
        for lag in sorted({0, 3, 6, 12, h["hac_lags"]}):
            hac_sensitivity.append({**meta, **hac(ref, cand, lag)})
        paired.append(pairs.assign(comparison_id=label))
    summary = pd.DataFrame(summaries)
    summary["hac_p_holm"] = np.nan
    for role in ["primary", "retrospective_robustness"]:
        mask = summary.role == role
        summary.loc[mask, "hac_p_holm"] = holm(summary.loc[mask, "hac_p_two_sided"])
    for name, frame in [("inference_summary", summary), ("bootstrap_sensitivity", pd.DataFrame(sensitivity)),
                        ("hac_sensitivity", pd.DataFrame(hac_sensitivity)), ("paired_losses", pd.concat(paired)),
                        ("crisis_resampling_groups", pd.DataFrame(crisis_groups)), ("macro_ar_bootstrap_draws", main_draws)]:
        frame.to_csv(output / f"{name}.csv", index=False)
    import json
    (output / "loss_distribution.json").write_text(json.dumps(distribution, indent=2) + "\n")
    figures(base, main_pairs, main_draws, output)
    print(summary[["comparison_id", "reduction", "hac_p_holm"]].to_string(index=False), flush=True)


def figures(base, pairs, draws, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".plot_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 2.8), layout="constrained")
    ax.plot(base.index, base.volatility*100, linewidth=.8)
    ax.set(xlabel="Month", ylabel="Annualized volatility (%)")
    ax.grid(alpha=.2)
    fig.savefig(output / "volatility.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 2.8), layout="constrained")
    ax.plot(pairs.target_date, pairs.loss_difference.cumsum()/pairs.reference_loss.sum()*100)
    ax.set(xlabel="Response month", ylabel="Cumulative gain (% of total AR loss)")
    ax.grid(alpha=.2)
    fig.savefig(output / "cumulative_gain.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    d = pairs.loss_difference.to_numpy()
    tolerance = 1e-12*max(pairs.reference_loss.mean(), pairs.candidate_loss.mean())
    active = np.abs(d)>tolerance
    axs[0, 0].stem(pairs.target_date, d*1e4, markerfmt=" ")
    axs[0, 0].set(title="Monthly loss differences", ylabel="Loss difference × 10⁴")
    axs[0, 1].bar(["Negative", "Zero", "Positive"], [(d < -tolerance).sum(), (~active).sum(), (d>tolerance).sum()])
    axs[0, 1].set(title="Signs and numerical zeros", ylabel="Months")
    axs[1, 0].bar(np.arange(active.sum()), np.sort(d[active])*1e4)
    axs[1, 0].set(title="Ordered nonzero differences", ylabel="Loss difference × 10⁴", xlabel="Observation rank")
    axs[1, 1].hist(draws.reduction*100, bins=50)
    axs[1, 1].axvline(0, color="black", linewidth=.8)
    axs[1, 1].set(title="Paired block-bootstrap reductions", xlabel="MSE reduction (%)", ylabel="Draws")
    fig.savefig(output / "loss_diagnostics.pdf")
    plt.close(fig)
