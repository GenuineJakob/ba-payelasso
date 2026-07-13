# src/train_paye_monthly_kitchen_sink.py

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
    predictors = AR6_PREDICTORS + predictor_columns

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

    model_comparison = evaluate_against_benchmark(
        actual=benchmark["actual"],
        benchmark=benchmark["benchmark"],
        predictions={
            "kitchen_sink_ols": ols_predictions,
            "kitchen_sink_lasso": lasso_predictions,
        },
    )
    model_comparison.insert(0, "predictor_set", "kitchen_sink")
    model_comparison.insert(1, "available_paye_predictors", ",".join(available.keys()))
    model_comparison.insert(2, "missing_paye_predictors", ",".join(missing["paye_predictor"]))

    predictions = pd.DataFrame(
        {
            "actual": benchmark["actual"],
            "benchmark": benchmark["benchmark"],
            "ols": ols_predictions,
            "lasso": lasso_predictions,
        }
    )

    model_comparison.to_csv(
        TABLES_DIR / "paye_monthly_kitchen_sink_model_comparison.csv",
        index=False,
    )
    predictions.to_csv(RESULTS_DIR / "paye_monthly_kitchen_sink_predictions.csv")
    ols_coefficients.to_csv(RESULTS_DIR / "paye_monthly_kitchen_sink_ols_coefficients.csv")
    lasso_coefficients.to_csv(RESULTS_DIR / "paye_monthly_kitchen_sink_lasso_coefficients.csv")
    missing.to_csv(TABLES_DIR / "paye_monthly_missing_predictors.csv", index=False)

    print("Paye monthly kitchen-sink model comparison:")
    print(model_comparison)
    print("\nMissing Paye predictors:")
    print(missing if not missing.empty else "None")


if __name__ == "__main__":
    main()
