"""Evaluate frozen FRED-MD pools with an unpenalized AR(6) block.

Usage: python src/train_fredmd.py --period main|holdout|all
Use --max-forecasts 3 for a short run, exported with a separate _smoke suffix.
"""

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

from build_fredmd_dataset import feature_pools, require_monthly_index
from download_fredmd import sha256_file
from fredmd_config import (
    ALPHAS, AR_CONTROLS, BUILD_MANIFEST_NAME, DATA_CAVEAT, DATASET_NAME,
    METADATA_NAME, PERIODS, POOL_NAMES, PROCESSED_DIR, RESULTS_DIR, TABLES_DIR,
    TARGET, VINTAGE, WINDOW_SIZE,
)
from partially_penalized_lasso import PartiallyPenalizedLassoCV, training_medians
from stable_ols import TrainingScaledOLS, OLS_CONFIGURATION, software_versions


def load_experiment(processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = processed_dir / BUILD_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError("Run python src/build_fredmd_dataset.py first.")
    manifest = json.loads(manifest_path.read_text())
    for key in ["dataset", "metadata"]:
        artifact = manifest[key]
        if sha256_file(processed_dir / artifact["file"]) != artifact["sha256"]:
            raise ValueError(f"Frozen {key} checksum mismatch; rebuild before forecasting.")
    if manifest["vintage"] != VINTAGE or manifest["window_size"] != WINDOW_SIZE:
        raise ValueError("Unexpected vintage or training-window specification.")
    data = pd.read_csv(processed_dir / DATASET_NAME, index_col=0, parse_dates=True)
    metadata = pd.read_csv(processed_dir / METADATA_NAME)
    require_monthly_index(data, "Frozen experiment dataset")
    if feature_pools(metadata) != manifest["pools"]:
        raise ValueError("Feature pools differ from the initial frozen selection.")
    return data, metadata, manifest


def fit_ols_window(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[float, pd.Series, dict]:
    """Fit an imputed OLS comparison without penalizing any coefficients."""
    predictors = AR_CONTROLS + features
    medians = training_medians(train[features])
    X = train[predictors].fillna(medians)
    forecast_X = test[predictors].fillna(medians)
    active = [feature for feature in features if X[feature].nunique() > 1]
    fitted_columns = AR_CONTROLS + active
    model = TrainingScaledOLS().fit(X[fitted_columns], train[TARGET])
    coefficients = pd.Series(0.0, index=["intercept"] + predictors)
    coefficients["intercept"] = model.intercept_
    coefficients.loc[fitted_columns] = model.coef_
    diagnostics = {
        **model.diagnostics(),
        "inactive_predictors": len(features) - len(active),
    }
    return float(model.predict(forecast_X[fitted_columns])[0]), coefficients, diagnostics


def evaluate_forecasts(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (period, pool, model), group in predictions.groupby(["period", "pool", "model"], sort=False):
        mse = float(np.mean((group.actual - group.prediction) ** 2))
        benchmark_mse = float(np.mean((group.actual - group.benchmark) ** 2))
        rows.append({
            "period": period, "pool": pool, "model": model, "n_forecasts": len(group),
            "first_origin": group.date.min(), "last_origin": group.date.max(),
            "n_predictors": int(group.n_predictors.iloc[0]), "MSE": mse,
            "pool_size": int(group.pool_size.iloc[0]), "n_ar_terms": len(AR_CONTROLS),
            "RMSE": np.sqrt(mse), "OOS_R2": 1 - mse / benchmark_mse,
        })
    return pd.DataFrame(rows)


def run_period(
    data: pd.DataFrame, metadata: pd.DataFrame, manifest: dict, period: str,
    results_dir: Path = RESULTS_DIR, tables_dir: Path = TABLES_DIR,
    max_forecasts: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if period not in PERIODS:
        raise ValueError(f"Unknown evaluation period: {period}")
    require_monthly_index(data, "Forecast dataset")
    start, end = map(pd.Timestamp, PERIODS[period])
    dates = data.loc[start:end].index
    expected = pd.date_range(start, end, freq="ME")
    if not dates.equals(expected):
        raise ValueError(f"The complete {period} evaluation period is not present in the dataset.")
    if max_forecasts is not None:
        if max_forecasts < 1:
            raise ValueError("--max-forecasts must be positive.")
        dates = dates[:max_forecasts]
    if not np.isfinite(data.loc[dates, [TARGET] + AR_CONTROLS]).all().all():
        raise ValueError("Forecast targets and AR controls must be complete for every pool.")
    pools = manifest["pools"]
    info = metadata.set_index("feature")
    prediction_rows, coefficient_rows, window_rows, imputation_rows = [], [], [], []
    quality_rows = []
    selections = {pool: [] for pool in POOL_NAMES}
    missing_counts = {pool: np.zeros(len(pools[pool]), dtype=int) for pool in POOL_NAMES}
    forecast_missing_counts = {pool: np.zeros(len(pools[pool]), dtype=int) for pool in POOL_NAMES}
    print(f"Running {period}: {len(dates)} origins, pools " + str({p: len(f) for p, f in pools.items()}), flush=True)

    # These small regressions are faster and more reproducible without spawning
    # a BLAS thread team for each training fold and alpha-path matrix product.
    with threadpool_limits(limits=1):
        for number, date in enumerate(dates, start=1):
            position = data.index.get_loc(date)
            if position < WINDOW_SIZE:
                raise ValueError(f"Not enough training history before {date.date()}.")
            train, test = data.iloc[position - WINDOW_SIZE:position], data.iloc[[position]]
            if not np.isfinite(train[[TARGET] + AR_CONTROLS]).all().all():
                raise ValueError("Training AR controls and targets must be finite; do not delete calendar rows.")
            ar = LinearRegression().fit(train[AR_CONTROLS], train[TARGET])
            benchmark = float(ar.predict(test[AR_CONTROLS])[0])
            actual = float(test[TARGET].iloc[0])
            for pool in POOL_NAMES:
                features = pools[pool]
                predictors = AR_CONTROLS + features
                raw_X, raw_test = train[features], test[features]
                medians = training_medians(raw_X)
                train_missing = raw_X.isna().sum()
                forecast_missing = raw_test.isna().iloc[0]
                missing_counts[pool] += train_missing.to_numpy()
                forecast_missing_counts[pool] += forecast_missing.to_numpy(dtype=int)
                for feature in features:
                    if train_missing[feature] or forecast_missing[feature]:
                        imputation_rows.append({
                            "date": date, "period": period, "pool": pool, "feature": feature,
                            "training_missing_n": int(train_missing[feature]),
                            "training_median": float(medians[feature]),
                            "forecast_imputed": bool(forecast_missing[feature]),
                        })
                ols_prediction, ols_coefficients, ols_diagnostics = fit_ols_window(train, test, features)
                lasso = PartiallyPenalizedLassoCV(
                    controls=AR_CONTROLS, alphas=ALPHAS, impute_missing=True, solver="lars"
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    try:
                        lasso.fit(train[predictors], train[TARGET])
                    except RuntimeError as exc:
                        raise RuntimeError(f"Failed at {date.date()}, pool={pool}: {exc}") from exc
                convergence_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
                other_warnings = [str(w.message) for w in caught if not issubclass(w.category, ConvergenceWarning)]
                lasso_prediction = float(lasso.predict(test[predictors])[0])
                lasso_coefficients = pd.Series(lasso.coef_, index=predictors)
                selected = lasso_coefficients.loc[features].abs() > 1e-12
                selections[pool].append(selected.to_numpy())
                common = {"date": date, "target_date": date + pd.offsets.MonthEnd(1),
                          "period": period, "pool": pool, "actual": actual,
                          "benchmark": benchmark, "pool_size": len(features)}
                for model, prediction, alpha, count in [
                    ("ar6_benchmark", benchmark, np.nan, 0),
                    ("ols", ols_prediction, np.nan, len(features)),
                    ("lasso", lasso_prediction, lasso.alpha_, int(selected.sum())),
                ]:
                    prediction_rows.append({**common, "model": model, "prediction": prediction,
                                            "alpha": alpha, "n_selected": count,
                                            "n_predictors": 0 if model == "ar6_benchmark" else len(features)})
                for model, coefficients, intercept, alpha in [
                    ("ols", ols_coefficients.drop("intercept"), ols_coefficients["intercept"], np.nan),
                    ("lasso", lasso_coefficients, lasso.intercept_, lasso.alpha_),
                ]:
                    coefficient_rows.append({"date": date, "period": period, "pool": pool,
                                             "model": model, "intercept": intercept,
                                             "alpha": alpha, **coefficients.to_dict()})
                window_rows.append({
                    "date": date, "period": period, "pool": pool,
                    "training_start": train.index[0], "training_end": train.index[-1],
                    "training_n": len(train), "n_predictors": len(features),
                    "training_imputed_fraction": float(train_missing.sum() / raw_X.size),
                    "forecast_imputed_fraction": float(forecast_missing.mean()),
                    "lasso_n_selected": int(selected.sum()), "lasso_alpha": lasso.alpha_,
                    "lasso_inactive_predictors": int((lasso.scaler_.var_ == 0).sum()),
                    "lasso_convergence_warnings": convergence_warnings,
                    "lasso_relative_dual_gap": lasso.relative_dual_gap_,
                    "lasso_selected_cv_relative_dual_gap": lasso.selected_cv_relative_dual_gap_,
                    "lasso_max_cv_relative_dual_gap": float(lasso.relative_dual_gap_path_.max()),
                    "lasso_polished_cv_alphas_n": lasso.polished_cv_alphas_n_,
                    "other_warnings": "; ".join(other_warnings), **ols_diagnostics,
                })
            if number == 1 or number % 12 == 0 or number == len(dates):
                print(f"  {period}: {number}/{len(dates)} origins through {date.date()}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    if not np.isfinite(predictions[["actual", "prediction", "benchmark"]]).all().all():
        raise ValueError("Nonfinite forecasts cannot be exported as valid results.")
    metrics = evaluate_forecasts(predictions)
    coefficients = pd.DataFrame(coefficient_rows)
    windows = pd.DataFrame(window_rows)
    selection_rows = []
    for pool in POOL_NAMES:
        feature_names = pools[pool]
        selected = np.vstack(selections[pool])
        for column, feature in enumerate(feature_names):
            details = info.loc[feature]
            selection_rows.append({
                "period": period, "pool": pool, "feature": feature,
                "source_code": details.source_code, "description": details.description,
                "group": int(details.group), "n_forecasts": len(dates),
                "selected_n": int(selected[:, column].sum()),
                "selected_share": float(selected[:, column].mean()),
            })
            quality_rows.append({
                "period": period, "pool": pool, "feature": feature,
                "reporting_lag_months": int(details.reporting_lag_months),
                "initial_coverage": float(details.initial_coverage),
                "mean_training_imputed_fraction": float(missing_counts[pool][column] / (WINDOW_SIZE * len(dates))),
                "forecast_imputed_n": int(forecast_missing_counts[pool][column]),
                "forecast_imputed_fraction": float(forecast_missing_counts[pool][column] / len(dates)),
            })
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    label = period + ("_smoke" if max_forecasts is not None else "")
    predictions.to_csv(results_dir / f"fredmd_predictions_{label}.csv", index=False)
    coefficients.to_csv(results_dir / f"fredmd_coefficients_{label}.csv", index=False)
    pd.DataFrame(imputation_rows, columns=[
        "date", "period", "pool", "feature", "training_missing_n", "training_median", "forecast_imputed"
    ]).to_csv(results_dir / f"fredmd_imputation_values_{label}.csv", index=False)
    metrics.to_csv(tables_dir / f"fredmd_model_comparison_{label}.csv", index=False)
    windows.to_csv(tables_dir / f"fredmd_window_diagnostics_{label}.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(tables_dir / f"fredmd_selection_frequency_{label}.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(tables_dir / f"fredmd_data_quality_{label}.csv", index=False)
    run_manifest = {
        "period": period, "smoke_run": max_forecasts is not None, "n_forecasts": len(dates),
        "vintage": VINTAGE, "caveat": DATA_CAVEAT, "window_size": WINDOW_SIZE,
        "n_splits": 5, "alphas": ALPHAS.tolist(), "impute_missing": True,
        "lasso_solver": "lars", "lasso_tolerance": 1e-6,
        "software_versions": software_versions(), "ols_configuration": OLS_CONFIGURATION,
        "first_origin": dates[0].strftime("%Y-%m-%d"), "last_origin": dates[-1].strftime("%Y-%m-%d"),
        "pool_sizes": {name: len(features) for name, features in pools.items()},
        "frozen_build": manifest, "lasso_convergence_warnings": int(windows.lasso_convergence_warnings.sum()),
    }
    (results_dir / f"fredmd_run_manifest_{label}.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(metrics.to_string(index=False), flush=True)
    if run_manifest["lasso_convergence_warnings"]:
        print("Convergence warnings recorded in window diagnostics:", run_manifest["lasso_convergence_warnings"], flush=True)
    return predictions, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", choices=["main", "holdout", "all"], default="main")
    parser.add_argument("--max-forecasts", type=int, help="Short run; uses separate _smoke output files.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    args = parser.parse_args()
    data, metadata, manifest = load_experiment(args.processed_dir)
    for period in PERIODS if args.period == "all" else [args.period]:
        run_period(data, metadata, manifest, period, args.results_dir, args.tables_dir, args.max_forecasts)
    print(DATA_CAVEAT)


if __name__ == "__main__":
    main()
