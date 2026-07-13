# src/train_paye_monthly_individual.py

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
    missing_paye_predictors,
    rolling_model_predictions,
)


def main() -> None:
    df = load_monthly_data()
    available = available_paye_predictors(df)
    missing = missing_paye_predictors(df)
    predictor_columns = list(available.values())

    sample = common_monthly_sample(df, predictor_columns)
    benchmark = get_or_create_ar6_benchmark(sample)

    metric_rows = []
    prediction_frames = []
    ols_coefficient_frames = []
    lasso_coefficient_frames = []

    for paye_name, column in available.items():
        predictors = AR6_PREDICTORS + [column]

        ols_predictions, ols_coefficients = rolling_model_predictions(
            df=sample,
            predictors=predictors,
            model_kind="ols",
        )
        lasso_predictions, lasso_coefficients = rolling_model_predictions(
            df=sample,
            predictors=predictors,
            model_kind="lasso",
        )

        metrics = evaluate_against_benchmark(
            actual=benchmark["actual"],
            benchmark=benchmark["benchmark"],
            predictions={
                f"{paye_name}_ols": ols_predictions,
                f"{paye_name}_lasso": lasso_predictions,
            },
        )
        metrics = metrics[metrics["model"] != "ar6_benchmark"].copy()
        metrics.insert(0, "paye_predictor", paye_name)
        metrics.insert(1, "dataset_column", column)
        metric_rows.append(metrics)

        predictions = pd.DataFrame(
            {
                "actual": benchmark["actual"],
                "benchmark": benchmark["benchmark"],
                "ols": ols_predictions,
                "lasso": lasso_predictions,
            }
        )
        predictions.insert(0, "paye_predictor", paye_name)
        prediction_frames.append(predictions.reset_index())

        ols_coefficients.insert(0, "paye_predictor", paye_name)
        lasso_coefficients.insert(0, "paye_predictor", paye_name)
        ols_coefficient_frames.append(ols_coefficients.reset_index())
        lasso_coefficient_frames.append(lasso_coefficients.reset_index())

    benchmark_metrics = evaluate_against_benchmark(
        actual=benchmark["actual"],
        benchmark=benchmark["benchmark"],
        predictions={},
    )
    benchmark_metrics.insert(0, "paye_predictor", "benchmark")
    benchmark_metrics.insert(1, "dataset_column", ",".join(AR6_PREDICTORS))

    model_comparison = pd.concat([benchmark_metrics] + metric_rows, ignore_index=True)
    model_comparison.to_csv(
        TABLES_DIR / "paye_monthly_individual_model_comparison.csv",
        index=False,
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "paye_monthly_individual_predictions.csv",
        index=False,
    )
    pd.concat(ols_coefficient_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "paye_monthly_individual_ols_coefficients.csv",
        index=False,
    )
    pd.concat(lasso_coefficient_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "paye_monthly_individual_lasso_coefficients.csv",
        index=False,
    )
    missing.to_csv(TABLES_DIR / "paye_monthly_missing_predictors.csv", index=False)

    print("Paye monthly individual model comparison:")
    print(model_comparison)
    print("\nMissing Paye predictors:")
    print(missing if not missing.empty else "None")


if __name__ == "__main__":
    main()
