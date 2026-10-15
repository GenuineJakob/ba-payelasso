"""Shared estimation and exports for the two separate volatility experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from build_volatility_analysis_datasets import SETTINGS
from build_fredmd_dataset import feature_pools
from download_fredmd import sha256_file
from fredmd_config import ALPHAS, PROCESSED_DIR, RESULTS_DIR, TABLES_DIR
from partially_penalized_lasso import PartiallyPenalizedLassoCV, training_medians


def require_calendar(data, kind):
    frequency = SETTINGS[kind]["frequency"]
    expected = pd.date_range(data.index[0], data.index[-1], freq=frequency)
    if not data.index.is_unique or not data.index.equals(expected):
        raise ValueError(f"{kind} requires unique consecutive calendar rows.")


def load_analysis(kind, processed_dir=PROCESSED_DIR):
    manifest_path = processed_dir / f"{kind}_build_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run python src/build_volatility_analysis_datasets.py first.")
    manifest = json.loads(manifest_path.read_text())
    if manifest["kind"] != kind:
        raise ValueError("Wrong experiment manifest.")
    for key in ["controls", "window_size", "frequency", "first_origin"]:
        if manifest[key] != SETTINGS[kind][key]:
            raise ValueError(f"Unexpected frozen setting: {key}")
    if manifest["periods"] != {key: list(value) for key, value in SETTINGS[kind]["periods"].items()}:
        raise ValueError("Unexpected evaluation periods.")
    for path, expected in manifest["sources"].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"Source changed; rebuild analysis datasets: {path}")
    for name in ["dataset", "metadata"]:
        info = manifest[name]
        if sha256_file(processed_dir / info["file"]) != info["sha256"]:
            raise ValueError(f"Frozen {name} checksum mismatch.")
    data = pd.read_csv(processed_dir / manifest["dataset"]["file"], index_col=0, parse_dates=True)
    data["target_date"] = pd.to_datetime(data.target_date)
    metadata = pd.read_csv(processed_dir / manifest["metadata"]["file"])
    require_calendar(data, kind)
    expected_pools = (feature_pools(metadata) if kind == "contemporaneous" else
                      {"current_quarterly": metadata.loc[metadata.eligible, "feature"].tolist()})
    if manifest["pools"] != expected_pools or any(not f for f in expected_pools.values()):
        raise ValueError("Frozen pools differ from eligible metadata.")
    return data, metadata, manifest


def scale_volatility(data, controls, scale):
    if scale not in {"log", "level"}:
        raise ValueError("Scale must be log or level.")
    result = data.copy()
    columns = ["target"] + controls
    values = result[columns].to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
        raise ValueError("Volatility must be positive and finite wherever observed; no offset is added.")
    if scale == "log":
        result[columns] = np.log(result[columns])
    return result


def fit_ols(train, test, controls, features):
    columns = controls + features
    medians = training_medians(train[features])
    X, test_X = train[columns].fillna(medians), test[columns].fillna(medians)
    active = [f for f in features if X[f].nunique() > 1]
    fitted_columns = controls + active
    model = LinearRegression().fit(X[fitted_columns], train.target)
    coefficients = pd.Series(0.0, index=columns)
    coefficients.loc[fitted_columns] = model.coef_
    return {
        "prediction": float(model.predict(test_X[fitted_columns])[0]),
        "coefficients": coefficients, "intercept": float(model.intercept_),
        "training_R2": float(model.score(X[fitted_columns], train.target)),
        "alpha": np.nan, "n_selected": len(active),
        "ols_rank": int(model.rank_), "ols_fitted_columns": len(fitted_columns),
        "ols_rank_deficient": bool(model.rank_ < len(fitted_columns)),
        "inactive_predictors": len(features) - len(active),
    }


def fit_lasso(train, test, controls, features):
    columns = controls + features
    model = PartiallyPenalizedLassoCV(
        controls=controls, alphas=ALPHAS, impute_missing=True, solver="lars",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train[columns], train.target)
    coefficients = pd.Series(model.coef_, index=columns)
    return {
        "prediction": float(model.predict(test[columns])[0]),
        "coefficients": coefficients, "intercept": float(model.intercept_),
        "training_R2": float(model.score(train[columns], train.target)),
        "alpha": float(model.alpha_),
        "n_selected": int((coefficients.loc[features].abs() > 1e-12).sum()),
        "inactive_predictors": int((model.scaler_.var_ == 0).sum()),
        "lasso_convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
        "lasso_relative_dual_gap": model.relative_dual_gap_,
        "lasso_selected_cv_relative_dual_gap": model.selected_cv_relative_dual_gap_,
        "lasso_max_cv_relative_dual_gap": float(model.relative_dual_gap_path_.max()),
        "lasso_polished_cv_alphas_n": model.polished_cv_alphas_n_,
        "other_warnings": "; ".join(str(w.message) for w in caught if not issubclass(w.category, ConvergenceWarning)),
    }


def evaluate_predictions(predictions):
    rows = []
    keys = ["experiment", "scale", "period", "pool", "model"]
    for group_key, group in predictions.groupby(keys, sort=False):
        mse = float(np.mean((group.actual - group.prediction) ** 2))
        ar_mse = float(np.mean((group.actual - group.benchmark) ** 2))
        reduction = 1 - mse / ar_mse if ar_mse > 0 else np.nan
        rows.append({**dict(zip(keys, group_key)), "n_evaluations": len(group),
                     "first_origin": group.date.min(), "last_origin": group.date.max(),
                     "first_target": group.target_date.min(), "last_target": group.target_date.max(),
                     "n_predictors": int(group.n_predictors.iloc[0]),
                     "n_ar_terms": int(group.n_ar_terms.iloc[0]),
                     "MSE": mse, "RMSE": np.sqrt(mse), "AR_MSE": ar_mse,
                     "error_reduction_vs_AR": reduction,
                     "OOS_R2": reduction if group_key[0] == "quarterly" else np.nan,
                     "heldout_explanatory_R2_vs_AR": reduction if group_key[0] == "contemporaneous" else np.nan,
                     "mean_training_R2": float(group.training_R2.mean()),
                     "diagnostic_only": bool(group.diagnostic_only.iloc[0])})
    return pd.DataFrame(rows)


def run_analysis(data, metadata, manifest, scale, period, results_dir=RESULTS_DIR,
                 tables_dir=TABLES_DIR, max_forecasts=None):
    kind = manifest["kind"]
    require_calendar(data, kind)
    settings = SETTINGS[kind]
    controls, window = settings["controls"], settings["window_size"]
    frame = scale_volatility(data, controls, scale)
    start, end = map(pd.Timestamp, settings["periods"][period])
    dates = frame.loc[start:end].index
    if not dates.equals(pd.date_range(start, end, freq=settings["frequency"])):
        raise ValueError("Incomplete evaluation calendar.")
    if max_forecasts is not None:
        if max_forecasts < 1:
            raise ValueError("--max-forecasts must be positive.")
        dates = dates[:max_forecasts]
    predictions, coefficient_rows, diagnostics, imputation_rows, quality_rows = [], [], [], [], []
    selections = {pool: [] for pool in manifest["pools"]}
    imputed = {pool: np.zeros(len(features)) for pool, features in manifest["pools"].items()}
    test_imputed = {pool: np.zeros(len(features)) for pool, features in manifest["pools"].items()}
    info = metadata.set_index("feature")
    label = f"{scale}_{period}" + ("_smoke" if max_forecasts is not None else "")
    print(f"{kind} {label}: {len(dates)} evaluations", flush=True)

    with threadpool_limits(limits=1):
        for number, date in enumerate(dates, 1):
            position = frame.index.get_loc(date)
            if position < window:
                raise ValueError("Insufficient training history.")
            train, test = frame.iloc[position-window:position], frame.iloc[[position]]
            if not np.isfinite(pd.concat([train, test])[["target"] + controls].to_numpy()).all():
                raise ValueError("AR controls and targets must be finite; calendar rows cannot be deleted.")
            if (train.target_date > date).any() or (kind == "contemporaneous" and (train.target_date >= date).any()):
                raise ValueError("Training responses are unavailable at the evaluation origin.")
            ar_fit = fit_ols(train, test, controls, [])
            benchmark = ar_fit["prediction"]
            for pool, features in manifest["pools"].items():
                medians = training_medians(train[features])
                missing, test_missing = train[features].isna().sum(), test[features].isna().iloc[0]
                imputed[pool] += missing.to_numpy()
                test_imputed[pool] += test_missing.to_numpy()
                for feature in features:
                    if missing[feature] or test_missing[feature]:
                        imputation_rows.append({"date": date, "scale": scale, "period": period,
                                                "pool": pool, "feature": feature,
                                                "training_missing_n": int(missing[feature]),
                                                "training_median": float(medians[feature]),
                                                "evaluation_imputed": bool(test_missing[feature])})
                fits = [(f"ar{len(controls)}", [], ar_fit),
                        ("joint_ols", features, fit_ols(train, test, controls, features)),
                        ("lasso", features, fit_lasso(train, test, controls, features))]
                if pool.startswith("current_"):
                    fits += [(f"individual__{f}", [f], fit_ols(train, test, controls, [f])) for f in features]
                for model_name, model_features, fitted in fits:
                    diagnostic_only = model_name == "joint_ols" and not pool.startswith("current_")
                    common = {"experiment": kind, "scale": scale, "period": period, "date": date,
                              "target_date": test.target_date.iloc[0], "pool": pool, "model": model_name}
                    predictions.append({**common, "actual": float(test.target.iloc[0]),
                                        "prediction": fitted["prediction"], "benchmark": benchmark,
                                        "n_predictors": len(model_features), "n_ar_terms": len(controls),
                                        "pool_size": len(features), "alpha": fitted["alpha"],
                                        "n_selected": fitted["n_selected"], "training_R2": fitted["training_R2"],
                                        "diagnostic_only": diagnostic_only})
                    for feature, coefficient in {"intercept": fitted["intercept"], **fitted["coefficients"].to_dict()}.items():
                        coefficient_rows.append({**common, "feature": feature, "coefficient": coefficient,
                                                 "penalized": model_name == "lasso" and feature in model_features})
                    diagnostics.append({**common, "training_start": train.index[0], "training_end": train.index[-1],
                                        "last_training_target": train.target_date.iloc[-1], "training_n": len(train),
                                        "diagnostic_only": diagnostic_only,
                                        "training_imputed_fraction": float(train[model_features].isna().to_numpy().mean()) if model_features else 0,
                                        "evaluation_imputed_fraction": float(test[model_features].isna().to_numpy().mean()) if model_features else 0,
                                        **{k: v for k, v in fitted.items() if k not in {"coefficients", "prediction", "intercept"}}})
                    if model_name == "lasso":
                        selections[pool].append((fitted["coefficients"].loc[features].abs() > 1e-12).to_numpy())
            if number == 1 or number % 12 == 0 or number == len(dates):
                print(f"  {number}/{len(dates)} through {date.date()}", flush=True)

    predictions = pd.DataFrame(predictions)
    if not np.isfinite(predictions[["actual", "prediction", "benchmark"]].to_numpy()).all():
        raise ValueError("Nonfinite predictions cannot be exported.")
    metrics = evaluate_predictions(predictions)
    selection_rows = []
    for pool, features in manifest["pools"].items():
        selected = np.vstack(selections[pool])
        for i, feature in enumerate(features):
            details = info.loc[feature]
            selection_rows.append({"scale": scale, "period": period, "pool": pool,
                                   "feature": feature, "description": details.description,
                                   "n_evaluations": len(dates), "selected_n": int(selected[:, i].sum()),
                                   "selected_share": float(selected[:, i].mean())})
            delay_key = "reporting_lag_months" if kind == "contemporaneous" else "reporting_lag_quarters"
            quality_rows.append({"scale": scale, "period": period, "pool": pool, "feature": feature,
                                 delay_key: int(details[delay_key]), "initial_coverage": float(details.initial_coverage),
                                 "mean_training_imputed_fraction": float(imputed[pool][i] / (window * len(dates))),
                                 "evaluation_imputed_n": int(test_imputed[pool][i]),
                                 "evaluation_imputed_fraction": float(test_imputed[pool][i] / len(dates))})
    diagnostics = pd.DataFrame(diagnostics)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(results_dir / f"{kind}_predictions_{label}.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(results_dir / f"{kind}_coefficients_{label}.csv", index=False)
    pd.DataFrame(imputation_rows, columns=["date", "scale", "period", "pool", "feature", "training_missing_n", "training_median", "evaluation_imputed"]).to_csv(
        results_dir / f"{kind}_imputation_values_{label}.csv", index=False)
    metrics.to_csv(tables_dir / f"{kind}_model_comparison_{label}.csv", index=False)
    diagnostics.to_csv(tables_dir / f"{kind}_window_diagnostics_{label}.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(tables_dir / f"{kind}_selection_frequency_{label}.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(tables_dir / f"{kind}_data_quality_{label}.csv", index=False)
    run_manifest = {"kind": kind, "scale": scale, "period": period, "smoke_run": max_forecasts is not None,
                    "n_evaluations": len(dates), "window_size": window, "n_splits": 5,
                    "alphas": ALPHAS.tolist(), "solver": "lars", "impute_missing": True,
                    "first_origin": dates[0].isoformat(), "last_origin": dates[-1].isoformat(),
                    "pool_sizes": {p: len(f) for p, f in manifest["pools"].items()},
                    "lasso_convergence_warnings": int(diagnostics.lasso_convergence_warnings.sum()),
                    "frozen_build": manifest}
    (results_dir / f"{kind}_run_manifest_{label}.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(metrics.loc[~metrics.model.str.startswith("individual"), ["pool", "model", "MSE", "error_reduction_vs_AR"]].to_string(index=False), flush=True)
    return predictions, metrics


def analysis_main(kind):
    parser = argparse.ArgumentParser(description=SETTINGS[kind]["interpretation"])
    parser.add_argument("--scale", choices=["log", "level", "both"], default="both")
    parser.add_argument("--period", choices=["main", "later", "all"], default="main")
    parser.add_argument("--max-forecasts", type=int)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()
    data, metadata, manifest = load_analysis(kind, args.processed_dir)
    scales = ["log", "level"] if args.scale == "both" else [args.scale]
    periods = list(SETTINGS[kind]["periods"]) if args.period == "all" else [args.period]
    for scale in scales:
        for period in periods:
            run_analysis(data, metadata, manifest, scale, period, args.results_dir, args.tables_dir, args.max_forecasts)
    print(manifest["interpretation"] + " " + manifest["caveat"])
