# src/train_models.py

"""
Train and evaluate volatility forecasting models.

This script supports two forecast designs:

1. Fixed split:
   - Train once on an early sample.
   - Forecast an out-of-sample test period.

2. Rolling window:
   - Re-estimate each model using the previous N months.
   - Forecast the next month.
   - Move the window forward by one month.

Models:
- Benchmark: AR(6), matching Paye's monthly lag specification.
- OLS: uses the AR(6) block and all macro predictors.
- Lasso: leaves the AR(6) block and intercept unpenalized; only macro
  predictors are subject to L1 regularization.

The target remains volatility in levels. Paye models log annualized
volatility; this script adopts his monthly lag order, not that transformation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from partially_penalized_lasso import PartiallyPenalizedLassoCV


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# CONFIG
# =====================================================

TRAIN_END = "2000-12-31"
TEST_START = "2001-01-01"
TEST_END = "2020-12-31"

TARGET = "target"

BENCHMARK_PREDICTORS = [
    "volatility",
] + [f"vol_lag_{lag}" for lag in range(1, 6)]

FULL_PREDICTORS = BENCHMARK_PREDICTORS + [
    "unemployment",
    "fed_funds",
    "default_spread",
    "term_spread",
    "commercial_paper_spread",
    "inflation",
    "ip_growth",
    "gdp_growth",
    "ppi_volatility",
    "ip_volatility",
]

# 360 months = 30 years of monthly data.
ROLLING_WINDOW_SIZE = 360

# A wider and explicit alpha grid makes Lasso tuning more transparent.
LASSO_ALPHAS = np.logspace(-5, 1, 200)


# =====================================================
# DATA LOADING AND VALIDATION
# =====================================================


def load_dataset() -> pd.DataFrame:
    """Load processed modelling dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run `python src/build_dataset.py` first."
        )

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Rows are forecast origins: target is Vol(t+1), so Vol(t), ..., Vol(t-5)
    # form the six autoregressive inputs. Build lags before dropping rows.
    for lag in range(1, 6):
        df[f"vol_lag_{lag}"] = df["volatility"].shift(lag)

    required_columns = set([TARGET] + BENCHMARK_PREDICTORS + FULL_PREDICTORS)
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    rows_before_drop = len(df)
    df = df.dropna(subset=list(required_columns))
    rows_dropped = rows_before_drop - len(df)

    if rows_dropped > 0:
        print(
            f"Dropped {rows_dropped} rows with missing modelling variables. "
            f"Usable sample starts at {df.index.min().date()}."
        )

    return df


# =====================================================
# EVALUATION
# =====================================================


def evaluate_predictions(
    y_true: pd.Series,
    predictions: pd.DataFrame,
    benchmark_column: str = "benchmark",
) -> pd.DataFrame:
    """
    Evaluate forecast accuracy using MSE, RMSE, and out-of-sample R².

    OOS_R2 is computed relative to the benchmark model:
        OOS_R2 = 1 - MSE_model / MSE_benchmark
    """
    if benchmark_column not in predictions.columns:
        raise ValueError(f"Benchmark column `{benchmark_column}` not found.")

    benchmark_mse = mean_squared_error(y_true, predictions[benchmark_column])

    rows = []
    for model_name in predictions.columns:
        mse = mean_squared_error(y_true, predictions[model_name])
        rmse = np.sqrt(mse)

        if model_name == benchmark_column:
            oos_r2 = np.nan
        else:
            oos_r2 = 1 - mse / benchmark_mse

        rows.append(
            {
                "model": model_name,
                "MSE": mse,
                "RMSE": rmse,
                "OOS_R2": oos_r2,
            }
        )

    return pd.DataFrame(rows).set_index("model")


# =====================================================
# MODEL FITTING HELPERS
# =====================================================


def fit_predict_benchmark(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit the unpenalized AR(6) benchmark and predict test observations."""
    model = LinearRegression()
    model.fit(train[BENCHMARK_PREDICTORS], train[TARGET])
    return model.predict(test[BENCHMARK_PREDICTORS])


def fit_predict_ols(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, LinearRegression]:
    """Fit OLS model with all predictors and predict test observations."""
    model = LinearRegression()
    model.fit(train[FULL_PREDICTORS], train[TARGET])
    predictions = model.predict(test[FULL_PREDICTORS])
    return predictions, model


def make_lasso_model(n_splits: int = 5) -> PartiallyPenalizedLassoCV:
    """Tune only the macro penalty, with fold-local projections and scaling."""
    return PartiallyPenalizedLassoCV(
        controls=BENCHMARK_PREDICTORS,
        alphas=LASSO_ALPHAS,
        n_splits=n_splits,
    )


def fit_predict_lasso(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, PartiallyPenalizedLassoCV]:
    """Fit Lasso model and predict test observations."""
    model = make_lasso_model()
    model.fit(train[FULL_PREDICTORS], train[TARGET])
    predictions = model.predict(test[FULL_PREDICTORS])
    return predictions, model


def get_lasso_coefficients(lasso_model: PartiallyPenalizedLassoCV) -> pd.Series:
    """Export the intercept and all coefficients in original feature units."""
    return pd.concat(
        [
            pd.Series({"intercept": lasso_model.intercept_}, name="coefficient"),
            pd.Series(
                lasso_model.coef_,
                index=lasso_model.feature_names_in_,
                name="coefficient",
            ),
        ]
    )


def selected_lasso_coefficients(coefficients: pd.Series) -> pd.Series:
    """Selection refers only to penalized predictors, never the AR block."""
    additional = coefficients.drop(["intercept"] + BENCHMARK_PREDICTORS)
    return additional[additional.abs() > 1e-12]


def get_ols_coefficients(ols_model: LinearRegression) -> pd.Series:
    """Extract OLS intercept and coefficients as a labelled Series."""
    coefficients = pd.Series(
        ols_model.coef_,
        index=FULL_PREDICTORS,
        name="coefficient",
    )
    return pd.concat(
        [
            pd.Series({"intercept": ols_model.intercept_}, name="coefficient"),
            coefficients,
        ]
    )


# =====================================================
# FIXED SPLIT FORECAST
# =====================================================


def fixed_split_forecast(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Run one fixed train/test split forecast experiment."""
    train = df.loc[:TRAIN_END]
    test = df.loc[TEST_START:TEST_END]

    if train.empty or test.empty:
        raise ValueError(
            "Fixed split produced an empty train or test set. "
            "Check TRAIN_END, TEST_START, and TEST_END."
        )

    print(f"Fixed split train period: {train.index.min().date()} to {train.index.max().date()}")
    print(f"Fixed split test period:  {test.index.min().date()} to {test.index.max().date()}")

    pred_benchmark = fit_predict_benchmark(train, test)
    pred_ols, ols_model = fit_predict_ols(train, test)
    pred_lasso, lasso_model = fit_predict_lasso(train, test)

    predictions = pd.DataFrame(
        {
            "actual": test[TARGET],
            "benchmark": pred_benchmark,
            "ols": pred_ols,
            "lasso": pred_lasso,
        },
        index=test.index,
    )

    results = evaluate_predictions(
        y_true=predictions["actual"],
        predictions=predictions[["benchmark", "ols", "lasso"]],
    )

    lasso_coefficients = get_lasso_coefficients(lasso_model)
    ols_coefficients = get_ols_coefficients(ols_model)

    return predictions, results, lasso_coefficients, ols_coefficients


# =====================================================
# ROLLING WINDOW FORECAST
# =====================================================


def rolling_window_forecast(
    df: pd.DataFrame,
    window_size: int = ROLLING_WINDOW_SIZE,
    start_date: str | None = TEST_START,
    end_date: str | None = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run rolling window forecasts.

    At each forecast date, the models are re-estimated using the previous
    `window_size` monthly observations. They then forecast the next observation.
    """
    if len(df) <= window_size:
        raise ValueError(
            f"Dataset has only {len(df)} observations, but window_size={window_size}."
        )

    prediction_rows = []
    lasso_coefficient_rows = []
    ols_coefficient_rows = []

    for i in range(window_size, len(df)):
        train = df.iloc[i - window_size : i]
        test = df.iloc[[i]]
        forecast_date = test.index[0]

        if start_date is not None and forecast_date < pd.Timestamp(start_date):
            continue
        if end_date is not None and forecast_date > pd.Timestamp(end_date):
            continue

        pred_benchmark = fit_predict_benchmark(train, test)[0]
        pred_ols, ols_model = fit_predict_ols(train, test)
        pred_ols = pred_ols[0]
        pred_lasso, lasso_model = fit_predict_lasso(train, test)
        pred_lasso = pred_lasso[0]

        lasso_coefficients = get_lasso_coefficients(lasso_model)
        ols_coefficients = get_ols_coefficients(ols_model)
        selected_variables = selected_lasso_coefficients(lasso_coefficients).index.tolist()

        prediction_rows.append(
            {
                "date": forecast_date,
                "actual": test[TARGET].iloc[0],
                "benchmark": pred_benchmark,
                "ols": pred_ols,
                "lasso": pred_lasso,
                "lasso_alpha": lasso_model.alpha_,
                "selected_variables": ",".join(selected_variables),
            }
        )

        lasso_coefficient_row = {"date": forecast_date}
        lasso_coefficient_row.update(lasso_coefficients.to_dict())
        lasso_coefficient_rows.append(lasso_coefficient_row)

        ols_coefficient_row = {"date": forecast_date}
        ols_coefficient_row.update(ols_coefficients.to_dict())
        ols_coefficient_rows.append(ols_coefficient_row)

    predictions = pd.DataFrame(prediction_rows).set_index("date")
    lasso_coefficients = pd.DataFrame(lasso_coefficient_rows).set_index("date")
    ols_coefficients = pd.DataFrame(ols_coefficient_rows).set_index("date")

    if predictions.empty:
        raise ValueError(
            "Rolling forecast produced no predictions. Check start_date, end_date, and window_size."
        )

    results = evaluate_predictions(
        y_true=predictions["actual"],
        predictions=predictions[["benchmark", "ols", "lasso"]],
    )

    return predictions, results, lasso_coefficients, ols_coefficients


# =====================================================
# MAIN
# =====================================================


def main() -> None:
    print("Loading dataset...")
    df = load_dataset()

    print("\nRunning fixed split forecast...")
    (
        fixed_predictions,
        fixed_results,
        fixed_lasso_coefficients,
        fixed_ols_coefficients,
    ) = fixed_split_forecast(df)

    print("\nFixed split results:")
    print(fixed_results)

    print("\nSelected Lasso variables in fixed split:")
    selected_fixed = selected_lasso_coefficients(fixed_lasso_coefficients)
    print(selected_fixed if not selected_fixed.empty else "No variables selected.")

    fixed_predictions.to_csv(RESULTS_DIR / "fixed_predictions.csv")
    fixed_results.to_csv(TABLES_DIR / "fixed_model_comparison.csv")
    fixed_lasso_coefficients.to_csv(RESULTS_DIR / "fixed_lasso_coefficients.csv")
    fixed_ols_coefficients.to_csv(RESULTS_DIR / "fixed_ols_coefficients.csv")

    print("\nRunning rolling window forecast...")
    (
        rolling_predictions,
        rolling_results,
        rolling_lasso_coefficients,
        rolling_ols_coefficients,
    ) = rolling_window_forecast(df)

    print("\nRolling window results:")
    print(rolling_results)

    rolling_predictions.to_csv(RESULTS_DIR / "rolling_predictions.csv")
    rolling_results.to_csv(TABLES_DIR / "rolling_model_comparison.csv")
    rolling_lasso_coefficients.to_csv(RESULTS_DIR / "rolling_lasso_coefficients.csv")
    rolling_ols_coefficients.to_csv(RESULTS_DIR / "rolling_ols_coefficients.csv")

    print("\nSaved model outputs:")
    print(f"- {RESULTS_DIR / 'fixed_predictions.csv'}")
    print(f"- {TABLES_DIR / 'fixed_model_comparison.csv'}")
    print(f"- {RESULTS_DIR / 'fixed_lasso_coefficients.csv'}")
    print(f"- {RESULTS_DIR / 'fixed_ols_coefficients.csv'}")
    print(f"- {RESULTS_DIR / 'rolling_predictions.csv'}")
    print(f"- {TABLES_DIR / 'rolling_model_comparison.csv'}")
    print(f"- {RESULTS_DIR / 'rolling_lasso_coefficients.csv'}")
    print(f"- {RESULTS_DIR / 'rolling_ols_coefficients.csv'}")


if __name__ == "__main__":
    main()
