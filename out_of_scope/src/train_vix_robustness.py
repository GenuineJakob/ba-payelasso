"""Matched same-month VIX robustness comparisons. No existing results change.

Usage: python src/train_vix_robustness.py --period all --scale both --window both
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from download_fredmd import sha256_file
from download_vix import FILENAME, MANIFEST, URL
from fredmd_config import RAW_DIR, RESULTS_DIR, TABLES_DIR, ALPHAS
from stable_ols import OLS_CONFIGURATION, software_versions
from volatility_analysis_utils import load_analysis, scale_volatility, fit_ols, fit_lasso, evaluate_predictions

VIX = "fredmd__VIXCLSx"
OPTIONS_VIX = "cboe__VIX"
CONTROLS = [f"vol_lag_{i}" for i in range(1, 7)]
DESIGNS = {120: {"main": ("2001-01-31", "2020-12-31"), "later": ("2021-01-31", "2023-11-30")},
           240: {"main": ("2010-01-31", "2020-12-31"), "later": ("2021-01-31", "2023-11-30")}}


def monthly_vix(daily):
    """Mean of daily CLOSE, not OHLC; missing months remain missing."""
    if not {"DATE", "CLOSE"}.issubset(daily):
        raise ValueError("Expected Cboe DATE and CLOSE columns.")
    dates = pd.to_datetime(daily.DATE, format="%m/%d/%Y", errors="raise")
    values = pd.to_numeric(daily.CLOSE, errors="raise").to_numpy(dtype=float)
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("VIX dates must be unique and increasing.")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("VIX closes must be positive and finite; no interpolation is permitted.")
    series = pd.Series(values, index=pd.DatetimeIndex(dates), name=OPTIONS_VIX)
    if series.index.min() < pd.Timestamp("1990-01-01"):
        raise ValueError("The modern options-based VIX source must not include a pre-1990 extension.")
    return series.resample("ME").mean()


def build_matched_data(data, pools, options):
    if set(pools["fredmd_full"]) - set(pools["fredmd_no_iv"]) != {VIX}:
        raise ValueError("Full and no-volatility-proxy pools must differ only by VIXCLSx.")
    result = data.copy()
    result[OPTIONS_VIX] = options.reindex(result.index)
    features = pools["fredmd_no_iv"]
    return result, {"no_volatility_proxy": features,
                    "extended_fredmd": pools["fredmd_full"],
                    "options_only": [OPTIONS_VIX if f == VIX else f for f in pools["fredmd_full"]]}


def matched_dates(data, window, period):
    start, end = map(pd.Timestamp, DESIGNS[window][period])
    expected = pd.date_range(start, end, freq="ME")
    if not data.loc[start:end].index.equals(expected):
        raise ValueError("Incomplete evaluation calendar.")
    first = data.index.get_loc(start)
    if first < window or data.index[first-window] < pd.Timestamp("1990-01-31"):
        raise ValueError("Insufficient genuine options-based training history.")
    history = data.iloc[first-window:data.index.get_loc(end)+1]
    if not np.isfinite(history[OPTIONS_VIX]).all():
        raise ValueError("Options VIX must be observed in every training/evaluation month; no imputation or backfill.")
    return expected


def history_audit(data):
    early = data.loc["1981-01-31":"1985-12-31", ["target", VIX]].dropna()
    rows = []
    for date in data.loc["2001-01-31":"2023-11-30"].index:
        pos = data.index.get_loc(date)
        train = data.iloc[pos-240:pos]
        rows.append({"date": date, "training_start": train.index[0], "training_end": train.index[-1],
                     "training_n": len(train), "synthetic_history_n": int((train.index < "1986-01-01").sum()),
                     "synthetic_history_fraction": float((train.index < "1986-01-01").mean())})
    return {"early_correlation": float(early.corr().iloc[0, 1]), "early_n": len(early),
            "early_period": "1981-01 through 1985-12", "evaluation_contains_pre1986_targets": False,
            "source_notes": {
                "fredmd_changes": "https://files.stlouisfed.org/files/htdocs/econ/mccracken/fred-databases/fredmdchanges.pdf",
                "updated_changes": "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/fredmdchanges0324.pdf?hash=4CC75807566F1F6F1BCC7FB312484BA0&sc_lang=en",
                "bloom_author_draft": "https://users.nber.org/~confer/2006/si2006/efg/bloom.pdf",
                "cboe": "https://www.cboe.com/tradable_products/vix/vix_historical_data/"},
            "interpretation": "Pre-1986 FRED-MD extension uses normalized realized-return volatility; later history contains options-based VXO/VIX. The change log does not specify every observation-level replacement boundary. Do not call the entire cached history observed implied volatility."}, pd.DataFrame(rows)


def run(data, pools, source_manifest, window, scale, period, max_forecasts=None,
        results_dir=RESULTS_DIR, tables_dir=TABLES_DIR):
    frame = scale_volatility(data, CONTROLS, scale)
    dates = matched_dates(frame, window, period)
    if max_forecasts is not None:
        if max_forecasts < 1:
            raise ValueError("--max-forecasts must be positive.")
        dates = dates[:max_forecasts]
    label = f"w{window}_{scale}_{period}" + ("_smoke" if max_forecasts else "")
    print(f"VIX robustness {label}: {len(dates)} months", flush=True)
    predictions, coefs, diagnostics, imputations, selections = [], [], [], [], []
    with threadpool_limits(limits=1):
        for number, date in enumerate(dates, 1):
            pos = frame.index.get_loc(date)
            train, test = frame.iloc[pos-window:pos], frame.iloc[[pos]]
            if not np.isfinite(pd.concat([train, test])[["target"]+CONTROLS].to_numpy()).all():
                raise ValueError("Targets and AR controls must be finite.")
            if not (train.target_date < date).all():
                raise ValueError("Training responses must precede the evaluation month.")
            ar = fit_ols(train, test, CONTROLS, [])
            fits = [("ar", "ar6", [], ar), ("options_only", "individual_vix_ols", [OPTIONS_VIX], fit_ols(train, test, CONTROLS, [OPTIONS_VIX]))]
            fits += [(pool, "lasso", features, fit_lasso(train, test, CONTROLS, features)) for pool, features in pools.items()]
            for pool, model, features, fitted in fits:
                common = {"experiment": "contemporaneous", "window": window, "scale": scale,
                          "period": period, "pool": pool, "model": model, "date": date,
                          "target_date": date}
                predictions.append({**common, "actual": float(test.target.iloc[0]),
                                    "prediction": fitted["prediction"], "benchmark": ar["prediction"],
                                    "n_predictors": len(features), "n_ar_terms": 6,
                                    "training_R2": fitted["training_R2"], "alpha": fitted["alpha"],
                                    "n_selected": fitted["n_selected"], "diagnostic_only": False})
                for feature, value in {"intercept": fitted["intercept"], **fitted["coefficients"].to_dict()}.items():
                    coefs.append({**common, "feature": feature, "coefficient": value,
                                  "penalized": model == "lasso" and feature in features})
                    if model == "lasso" and feature in features:
                        selections.append({"pool": pool, "feature": feature, "selected": abs(value) > 1e-12})
                diagnostics.append({**common, "training_start": train.index[0], "training_end": train.index[-1],
                                    "training_n": len(train), "synthetic_history_n": int((train.index < "1986-01-01").sum()),
                                    "training_imputed_fraction": float(train[features].isna().to_numpy().mean()) if features else 0,
                                    **{k: v for k, v in fitted.items() if k not in {"coefficients", "prediction", "intercept"}}})
                if features:
                    from partially_penalized_lasso import training_medians
                    medians = training_medians(train[features])
                    for feature in features:
                        missing = int(train[feature].isna().sum())
                        if missing or pd.isna(test[feature].iloc[0]):
                            imputations.append({**common, "feature": feature, "training_missing_n": missing,
                                                "training_median": medians[feature], "evaluation_imputed": pd.isna(test[feature].iloc[0])})
            if number == 1 or number % 12 == 0 or number == len(dates):
                print(f"  {number}/{len(dates)} through {date.date()}", flush=True)
    predictions = pd.DataFrame(predictions)
    if not np.isfinite(predictions[["actual", "prediction", "benchmark"]]).all().all():
        raise ValueError("Nonfinite results.")
    metrics = evaluate_predictions(predictions)
    metrics["window"] = window
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, output in [("predictions", predictions), ("coefficients", pd.DataFrame(coefs)),
                         ("imputation_values", pd.DataFrame(imputations))]:
        output.to_csv(results_dir / f"vix_robustness_{name}_{label}.csv", index=False)
    metrics.to_csv(tables_dir / f"vix_robustness_model_comparison_{label}.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(tables_dir / f"vix_robustness_window_diagnostics_{label}.csv", index=False)
    frequency = pd.DataFrame(selections).groupby(["pool", "feature"]).selected.agg(["sum", "mean", "count"]).reset_index()
    frequency.rename(columns={"sum": "selected_n", "mean": "selected_share", "count": "n_evaluations"}).to_csv(
        tables_dir / f"vix_robustness_selection_frequency_{label}.csv", index=False)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "window": window, "scale": scale,
                "period": period, "n_evaluations": len(dates), "smoke_run": max_forecasts is not None,
                "pools": pools, "controls": CONTROLS, "sources": source_manifest,
                "alphas": ALPHAS.tolist(), "n_splits": 5, "solver": "lars", "software_versions": software_versions(),
                "ols_configuration": OLS_CONFIGURATION, "predictor_eligibility": "Existing frozen 1981-2000 pools retained; only VIX series replaced. No test-period reselection.",
                "options_vix": "Arithmetic monthly mean of Cboe daily CLOSE. No pre-1990 extension or imputation.",
                "interpretation": "Ex-post explanation, not a forecast or causal effect. Later period is retrospective robustness."}
    (results_dir / f"vix_robustness_run_manifest_{label}.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["log", "level", "both"], default="both")
    parser.add_argument("--period", choices=["main", "later", "all"], default="main")
    parser.add_argument("--window", choices=["120", "240", "both"], default="both")
    parser.add_argument("--max-forecasts", type=int)
    args = parser.parse_args()
    source = json.loads((RAW_DIR / MANIFEST).read_text())
    if source["url"] != URL or sha256_file(RAW_DIR / FILENAME) != source["sha256"]:
        raise ValueError("Options VIX cache does not match its source manifest.")
    data, _, build = load_analysis("contemporaneous")
    options = monthly_vix(pd.read_csv(RAW_DIR / FILENAME))
    audit, exposure = history_audit(data)
    audit["cached_fredmd_first_observation"] = str(data[VIX].first_valid_index().date())
    # The joined target dataset begins in 1971; report the actual raw series start separately.
    from build_fredmd_dataset import read_snapshot
    from fredmd_config import SNAPSHOT_NAME
    raw, _ = read_snapshot(RAW_DIR / SNAPSHOT_NAME)
    audit["cached_fredmd_first_observation"] = str(raw.VIXCLSx.first_valid_index().date())
    audit["options_source"] = source
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "vix_history_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    exposure.to_csv(TABLES_DIR / "vix_synthetic_training_exposure.csv", index=False)
    data, pools = build_matched_data(data, build["pools"], options)
    sources = {"original_build_manifest": build, "cboe": source}
    for window in ([120, 240] if args.window == "both" else [int(args.window)]):
        for scale in (["log", "level"] if args.scale == "both" else [args.scale]):
            for period in (["main", "later"] if args.period == "all" else [args.period]):
                run(data, pools, sources, window, scale, period, args.max_forecasts)


if __name__ == "__main__":
    main()
