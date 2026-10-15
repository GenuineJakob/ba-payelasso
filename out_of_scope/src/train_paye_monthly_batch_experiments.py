# src/train_paye_monthly_batch_experiments.py

import numpy as np
import pandas as pd

from paye_monthly_utils import (
    AR6_PREDICTORS,
    RESULTS_DIR,
    TABLES_DIR,
    available_paye_predictors,
    common_monthly_sample,
    evaluate_against_benchmark,
    get_or_create_ar6_benchmark,
    load_monthly_data,
    rolling_model_predictions,
)


EXPERIMENTS = {
    "paye_replication": {
        "window": 240,
        "forecast_start": "1993-01-31",
        "forecast_end": "2010-12-31",
    },
    "modern_extension": {
        "window": 240,
        "forecast_start": "2001-01-31",
        "forecast_end": "2019-12-31",
    },
}

ALPHA_GRIDS = {
    #"wide_1e-5_1e1": np.logspace(-5, 1, 200),
    "moderate_1e-4_1e0": np.logspace(-4, 0, 200),
    "strong_1e-3_1e0": np.logspace(-3, 0, 200),
}

PREDICTOR_BLOCKS = {
    "best_individuals": ["dfy", "ipvol", "npy"],
    "best_individuals_plus_ip": ["dfy", "ipvol", "npy", "ip"],
    "credit": ["cp", "dfr", "dfy"],
    "real_activity": ["ip", "ipvol", "ppivol"],
    "return_payout": ["exret", "npy"],
    "credit_plus_activity": ["cp", "dfy", "ip", "ipvol"],
    "all_available": ["cp", "dfr", "dfy", "exret", "ip", "ipvol", "npy", "ppivol", "tms"],
}


def resolve_predictor_block(block_names: list[str], available: dict[str, str]) -> tuple[list[str], list[str]]:
    used_names = [name for name in block_names if name in available]
    columns = [available[name] for name in used_names]
    return used_names, columns


def common_experiment_sample(df: pd.DataFrame, available: dict[str, str]) -> pd.DataFrame:
    predictor_names = sorted(
        {
            predictor_name
            for block_predictor_names in PREDICTOR_BLOCKS.values()
            for predictor_name in block_predictor_names
            if predictor_name in available
        }
    )
    predictor_columns = [available[name] for name in predictor_names]
    return common_monthly_sample(df, predictor_columns)


def run_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    block_name: str,
    block_predictor_names: list[str],
    window_size: int,
    forecast_start: str,
    forecast_end: str,
    alpha_grid_name: str,
    alphas: np.ndarray,
    available: dict[str, str],
    include_baseline_models: bool,
) -> tuple[list[dict], pd.DataFrame]:
    used_names, predictor_columns = resolve_predictor_block(block_predictor_names, available)

    if not predictor_columns:
        return [], pd.DataFrame()

    sample = df
    if len(sample) <= window_size:
        return [], pd.DataFrame()

    benchmark = get_or_create_ar6_benchmark(
        sample,
        window_size=window_size,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        cache_label=experiment_name,
    )
    predictors = AR6_PREDICTORS + predictor_columns

    model_predictions = {}
    if include_baseline_models:
        ols_predictions, _ = rolling_model_predictions(
            df=sample,
            predictors=predictors,
            model_kind="ols",
            window_size=window_size,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
        )
        model_predictions["ols"] = ols_predictions

    lasso_predictions, lasso_coefficients = rolling_model_predictions(
        df=sample,
        predictors=predictors,
        model_kind="lasso",
        window_size=window_size,
        lasso_alphas=alphas,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )
    model_predictions["lasso"] = lasso_predictions

    metrics = evaluate_against_benchmark(
        actual=benchmark["actual"],
        benchmark=benchmark["benchmark"],
        predictions=model_predictions,
    )
    if not include_baseline_models:
        metrics = metrics[metrics["model"] != "ar6_benchmark"]

    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            {
                "experiment": experiment_name,
                "block": block_name,
                "predictors": ",".join(used_names),
                "model": row["model"],
                "window_size": window_size,
                "alpha_grid": alpha_grid_name if row["model"] == "lasso" else "",
                "MSE": row["MSE"],
                "RMSE": row["RMSE"],
                "OOS_R2": row["OOS_R2"],
                "n_forecasts": len(benchmark),
                "sample_start": sample.index.min().date(),
                "forecast_start": benchmark.index.min().date(),
                "forecast_end": benchmark.index.max().date(),
            }
        )

    prediction_columns = {"actual": benchmark["actual"], "lasso": lasso_predictions}
    if include_baseline_models:
        prediction_columns["benchmark"] = benchmark["benchmark"]
        prediction_columns["ols"] = ols_predictions

    predictions = pd.DataFrame(prediction_columns).reset_index()
    predictions.insert(1, "block", block_name)
    predictions.insert(2, "predictors", ",".join(used_names))
    predictions.insert(1, "experiment", experiment_name)
    predictions.insert(4, "window_size", window_size)
    predictions.insert(5, "alpha_grid", alpha_grid_name)

    lasso_coefficients = lasso_coefficients.reset_index()
    lasso_coefficients.insert(1, "block", block_name)
    lasso_coefficients.insert(2, "predictors", ",".join(used_names))
    lasso_coefficients.insert(1, "experiment", experiment_name)
    lasso_coefficients.insert(4, "window_size", window_size)
    lasso_coefficients.insert(5, "alpha_grid", alpha_grid_name)
    lasso_coefficients.to_csv(
        RESULTS_DIR
        / (
            "paye_monthly_batch_lasso_coefficients_"
            f"{experiment_name}_{block_name}_{window_size}_{alpha_grid_name}.csv"
        ),
        index=False,
    )

    return rows, predictions


def main() -> None:
    df = load_monthly_data()
    available = available_paye_predictors(df)
    df = common_experiment_sample(df, available)

    all_rows = []
    prediction_frames = []

    for experiment_name, experiment in EXPERIMENTS.items():
        window_size = experiment["window"]
        forecast_start = experiment["forecast_start"]
        forecast_end = experiment["forecast_end"]

        for block_name, block_predictor_names in PREDICTOR_BLOCKS.items():
            first_alpha_grid = True
            for alpha_grid_name, alphas in ALPHA_GRIDS.items():
                print(
                    (
                        f"Running experiment={experiment_name}, block={block_name}, "
                        f"window={window_size}, alpha_grid={alpha_grid_name}"
                    ),
                    flush=True,
                )
                rows, predictions = run_experiment(
                    df=df,
                    experiment_name=experiment_name,
                    block_name=block_name,
                    block_predictor_names=block_predictor_names,
                    window_size=window_size,
                    forecast_start=forecast_start,
                    forecast_end=forecast_end,
                    alpha_grid_name=alpha_grid_name,
                    alphas=alphas,
                    available=available,
                    include_baseline_models=first_alpha_grid,
                )
                all_rows.extend(rows)
                if not predictions.empty:
                    prediction_frames.append(predictions)
                first_alpha_grid = False

    results = pd.DataFrame(all_rows)
    if results.empty:
        raise RuntimeError("No batch experiment results were produced.")

    results = results.sort_values(["experiment", "OOS_R2", "RMSE"], ascending=[True, False, True])
    results.to_csv(TABLES_DIR / "paye_monthly_batch_experiment_results.csv", index=False)

    best_by_model = (
        results[results["model"] != "ar6_benchmark"]
        .sort_values(["experiment", "model", "RMSE"])
        .groupby(["experiment", "model"], as_index=False)
        .head(10)
        .sort_values(["experiment", "model", "RMSE"])
    )
    best_by_model.to_csv(TABLES_DIR / "paye_monthly_batch_best_by_model.csv", index=False)

    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            RESULTS_DIR / "paye_monthly_batch_predictions.csv",
            index=False,
        )

    print("\nBest batch results by OOS_R2:")
    print(
        results.sort_values(["experiment", "OOS_R2", "RMSE"], ascending=[True, False, True])
        .groupby("experiment", as_index=False)
        .head(10)
        .to_string(index=False)
    )
    print("\nSaved:")
    print(f"- {TABLES_DIR / 'paye_monthly_batch_experiment_results.csv'}")
    print(f"- {TABLES_DIR / 'paye_monthly_batch_best_by_model.csv'}")
    print(f"- {RESULTS_DIR / 'paye_monthly_batch_predictions.csv'}")


if __name__ == "__main__":
    main()
