# src/evaluate_paye_monthly_outputs.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paye_monthly_utils import AR6_PREDICTORS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

BATCH_RESULTS_PATH = TABLES_DIR / "paye_monthly_batch_experiment_results.csv"
BEST_BY_MODEL_PATH = TABLES_DIR / "paye_monthly_batch_best_by_model.csv"
INDIVIDUAL_RESULTS_PATH = TABLES_DIR / "paye_monthly_individual_model_comparison.csv"
KITCHEN_RESULTS_PATH = TABLES_DIR / "paye_monthly_kitchen_sink_model_comparison.csv"
MISSING_PREDICTORS_PATH = TABLES_DIR / "paye_monthly_missing_predictors.csv"

INSIGHT_SUMMARY_PATH = TABLES_DIR / "paye_monthly_output_insights.csv"
BATCH_BEST_PATH = TABLES_DIR / "paye_monthly_batch_best_insights.csv"
LASSO_SELECTION_PATH = TABLES_DIR / "paye_monthly_lasso_selection_frequency.csv"
FILE_INVENTORY_PATH = TABLES_DIR / "paye_monthly_output_file_inventory.csv"

METADATA_COLUMNS = {
    "date",
    "experiment",
    "block",
    "predictors",
    "window_size",
    "alpha_grid",
    "intercept",
    "alpha",
}


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def file_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(PROJECT_ROOT.joinpath("output").glob("*/*.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
            rows.append(
                {
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "columns": len(header.columns),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file": str(path.relative_to(PROJECT_ROOT)),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "columns": np.nan,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def add_benchmark_gap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "experiment" not in df.columns:
        return df

    output = df.copy()
    benchmark = (
        output[output["model"] == "ar6_benchmark"]
        .drop_duplicates(subset=["experiment"])
        .set_index("experiment")[["MSE", "RMSE"]]
        .rename(columns={"MSE": "benchmark_MSE", "RMSE": "benchmark_RMSE"})
    )
    output = output.join(benchmark, on="experiment")
    output["MSE_gap_vs_benchmark"] = output["MSE"] - output["benchmark_MSE"]
    output["RMSE_gap_vs_benchmark"] = output["RMSE"] - output["benchmark_RMSE"]
    return output


def best_batch_insights(batch: pd.DataFrame) -> pd.DataFrame:
    if batch.empty:
        return pd.DataFrame()

    batch = add_benchmark_gap(batch)
    candidates = batch[batch["model"] != "ar6_benchmark"].copy()
    if candidates.empty:
        return pd.DataFrame()

    best_overall = (
        candidates.sort_values(["experiment", "RMSE"])
        .groupby("experiment", as_index=False)
        .head(1)
        .assign(insight="best_non_benchmark")
    )
    best_lasso = (
        candidates[candidates["model"] == "lasso"]
        .sort_values(["experiment", "RMSE"])
        .groupby("experiment", as_index=False)
        .head(1)
        .assign(insight="best_lasso")
    )
    best_ols = (
        candidates[candidates["model"] == "ols"]
        .sort_values(["experiment", "RMSE"])
        .groupby("experiment", as_index=False)
        .head(1)
        .assign(insight="best_ols")
    )

    return pd.concat([best_overall, best_lasso, best_ols], ignore_index=True).sort_values(
        ["experiment", "insight"]
    )


def lasso_selection_frequency() -> pd.DataFrame:
    rows = []
    coefficient_files = sorted(RESULTS_DIR.glob("paye_monthly_batch_lasso_coefficients_*.csv"))

    for path in coefficient_files:
        df = pd.read_csv(path)
        if df.empty:
            continue

        coefficient_columns = [
            col
            for col in df.columns
            if col not in METADATA_COLUMNS
            and col not in AR6_PREDICTORS
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        if not coefficient_columns:
            continue

        for column in coefficient_columns:
            nonzero = df[column].abs() > 1e-12
            rows.append(
                {
                    "file": path.name,
                    "experiment": df.get("experiment", pd.Series(["unknown"])).iloc[0],
                    "block": df.get("block", pd.Series(["unknown"])).iloc[0],
                    "alpha_grid": df.get("alpha_grid", pd.Series(["unknown"])).iloc[0],
                    "coefficient": column,
                    "nonzero_share": nonzero.mean(),
                    "mean_abs_coefficient": df[column].abs().mean(),
                    "mean_coefficient": df[column].mean(),
                }
            )

    if not rows:
        return pd.DataFrame()

    selection = pd.DataFrame(rows)
    return selection.sort_values(
        ["experiment", "block", "alpha_grid", "nonzero_share", "mean_abs_coefficient"],
        ascending=[True, True, True, False, False],
    )


def headline_summary(batch: pd.DataFrame) -> pd.DataFrame:
    if batch.empty:
        return pd.DataFrame()

    batch = add_benchmark_gap(batch)
    rows = []
    for experiment, group in batch.groupby("experiment"):
        benchmark = group[group["model"] == "ar6_benchmark"].head(1)
        candidates = group[group["model"] != "ar6_benchmark"]
        best = candidates.sort_values("RMSE").head(1)
        best_lasso = candidates[candidates["model"] == "lasso"].sort_values("RMSE").head(1)
        best_ols = candidates[candidates["model"] == "ols"].sort_values("RMSE").head(1)

        row = {
            "experiment": experiment,
            "benchmark_RMSE": benchmark["RMSE"].iloc[0] if not benchmark.empty else np.nan,
            "benchmark_MSE": benchmark["MSE"].iloc[0] if not benchmark.empty else np.nan,
            "best_model": best["model"].iloc[0] if not best.empty else "",
            "best_block": best["block"].iloc[0] if not best.empty else "",
            "best_RMSE": best["RMSE"].iloc[0] if not best.empty else np.nan,
            "best_OOS_R2": best["OOS_R2"].iloc[0] if not best.empty else np.nan,
            "best_lasso_block": best_lasso["block"].iloc[0] if not best_lasso.empty else "",
            "best_lasso_RMSE": best_lasso["RMSE"].iloc[0] if not best_lasso.empty else np.nan,
            "best_lasso_OOS_R2": best_lasso["OOS_R2"].iloc[0] if not best_lasso.empty else np.nan,
            "best_ols_block": best_ols["block"].iloc[0] if not best_ols.empty else "",
            "best_ols_RMSE": best_ols["RMSE"].iloc[0] if not best_ols.empty else np.nan,
            "best_ols_OOS_R2": best_ols["OOS_R2"].iloc[0] if not best_ols.empty else np.nan,
            "n_forecasts": benchmark["n_forecasts"].iloc[0] if not benchmark.empty else np.nan,
            "forecast_start": benchmark["forecast_start"].iloc[0] if not benchmark.empty else "",
            "forecast_end": benchmark["forecast_end"].iloc[0] if not benchmark.empty else "",
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_table(title: str, df: pd.DataFrame, columns: list[str] | None = None, rows: int = 12) -> None:
    print(f"\n{title}")
    if df.empty:
        print("No data.")
        return
    display = df[columns] if columns else df
    print(display.head(rows).to_string(index=False))


def main() -> None:
    batch = read_csv_if_exists(BATCH_RESULTS_PATH)
    individual = read_csv_if_exists(INDIVIDUAL_RESULTS_PATH)
    kitchen = read_csv_if_exists(KITCHEN_RESULTS_PATH)
    missing = read_csv_if_exists(MISSING_PREDICTORS_PATH)

    inventory = file_inventory()
    summary = headline_summary(batch)
    batch_best = best_batch_insights(batch)
    selection = lasso_selection_frequency()

    inventory.to_csv(FILE_INVENTORY_PATH, index=False)
    if not summary.empty:
        summary.to_csv(INSIGHT_SUMMARY_PATH, index=False)
    if not batch_best.empty:
        batch_best.to_csv(BATCH_BEST_PATH, index=False)
    if not selection.empty:
        selection.to_csv(LASSO_SELECTION_PATH, index=False)

    print_table("Headline summary", summary)
    print_table(
        "Best batch models",
        batch_best,
        [
            "experiment",
            "insight",
            "model",
            "block",
            "alpha_grid",
            "RMSE",
            "OOS_R2",
            "RMSE_gap_vs_benchmark",
            "n_forecasts",
        ],
    )
    print_table(
        "Most frequently selected Lasso coefficients",
        selection,
        [
            "experiment",
            "block",
            "alpha_grid",
            "coefficient",
            "nonzero_share",
            "mean_abs_coefficient",
        ],
    )
    print_table("Missing predictors", missing)

    if not individual.empty:
        print_table(
            "Best individual models",
            individual.sort_values("RMSE"),
            ["paye_predictor", "model", "RMSE", "OOS_R2"],
        )

    if not kitchen.empty:
        print_table(
            "Kitchen sink models",
            kitchen.sort_values("RMSE"),
            ["predictor_set", "model", "RMSE", "OOS_R2"],
        )

    print("\nSaved insight files:")
    for path in [
        INSIGHT_SUMMARY_PATH,
        BATCH_BEST_PATH,
        LASSO_SELECTION_PATH,
        FILE_INVENTORY_PATH,
    ]:
        if path.exists():
            print(f"- {path}")


if __name__ == "__main__":
    main()
