# src/paye_monthly_utils.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from partially_penalized_lasso import PartiallyPenalizedLassoCV, training_medians


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target"
TEST_START = "2001-01-01"
TEST_END = "2020-12-31"
ROLLING_WINDOW_SIZE = 240
LASSO_ALPHAS = np.logspace(-5, 1, 200)

# Paye uses AR(6) for monthly data and AR(2) for quarterly data. We retain
# volatility in levels here; his main specifications use log volatility.
AR6_PREDICTORS = ["vol"] + [f"vol_lag_{lag}" for lag in range(1, 6)]

PAYE_MONTHLY_PREDICTORS = {
    "cp": "commercial_paper_spread",
    "dfr": "dfr",
    "dfy": "default_spread",
    "exret": "exret",
    "ip": "ip_growth",
    "ipvol": "ip_volatility",
    "npy": "npy",
    "ppivol": "ppi_volatility",
    "tms": "term_spread",
}

AR6_BENCHMARK_PATH = RESULTS_DIR / "paye_monthly_ar6_benchmark_predictions.csv"


def forecast_period_label(
    forecast_start: str | pd.Timestamp,
    forecast_end: str | pd.Timestamp,
) -> str:
    start = pd.Timestamp(forecast_start).strftime("%Y%m%d")
    end = pd.Timestamp(forecast_end).strftime("%Y%m%d")
    return f"{start}_{end}"


def load_monthly_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run `python src/build_dataset.py` first."
        )

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()

    df["vol"] = df["volatility"]
    for lag in range(1, 6):
        df[f"vol_lag_{lag}"] = df["volatility"].shift(lag)

    return df


def available_paye_predictors(df: pd.DataFrame) -> dict[str, str]:
    return {
        paye_name: column
        for paye_name, column in PAYE_MONTHLY_PREDICTORS.items()
        if column in df.columns
    }


def missing_paye_predictors(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"paye_predictor": paye_name, "expected_column": column}
        for paye_name, column in PAYE_MONTHLY_PREDICTORS.items()
        if column not in df.columns
    ]
    return pd.DataFrame(rows, columns=["paye_predictor", "expected_column"])


def common_monthly_sample(df: pd.DataFrame, predictor_columns: list[str]) -> pd.DataFrame:
    required_columns = [TARGET] + AR6_PREDICTORS + predictor_columns
    return df.dropna(subset=required_columns)


def make_lasso_model(
    alphas: np.ndarray | None = None, impute_missing: bool = False
) -> PartiallyPenalizedLassoCV:
    """Create the shared Lasso with the full AR(6) block unpenalized."""
    if alphas is None:
        alphas = LASSO_ALPHAS

    return PartiallyPenalizedLassoCV(
        controls=AR6_PREDICTORS, alphas=alphas, impute_missing=impute_missing
    )


def fit_predict_lasso_with_unpenalized_controls(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictors: list[str],
    alphas: np.ndarray | None = None,
    impute_missing: bool = False,
) -> tuple[float, dict[str, float]]:
    """Fit a partially penalized Lasso with the AR(6) block unpenalized.

    The shared estimator residualizes the target and additional predictors
    against AR controls within each training fold, penalizes only additional
    predictors, and recovers the intercept and AR coefficients by OLS.
    """
    if not set(AR6_PREDICTORS).issubset(predictors):
        raise ValueError(
            "Lasso models must contain the complete AR(6) block as unpenalized controls."
        )
    lasso_model = make_lasso_model(alphas=alphas, impute_missing=impute_missing)
    lasso_model.fit(train[predictors], train[TARGET])
    prediction = lasso_model.predict(test[predictors])[0]
    coefficients = {
        "intercept": lasso_model.intercept_,
        "alpha": lasso_model.alpha_,
    }
    coefficients.update(dict(zip(predictors, lasso_model.coef_)))
    return float(prediction), coefficients


def rolling_model_predictions(
    df: pd.DataFrame,
    predictors: list[str],
    model_kind: str,
    window_size: int = ROLLING_WINDOW_SIZE,
    lasso_alphas: np.ndarray | None = None,
    forecast_start: str | pd.Timestamp = TEST_START,
    forecast_end: str | pd.Timestamp = TEST_END,
    impute_missing: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    prediction_rows = []
    coefficient_rows = []
    forecast_start = pd.Timestamp(forecast_start)
    forecast_end = pd.Timestamp(forecast_end)

    for i in range(window_size, len(df)):
        train = df.iloc[i - window_size : i]
        test = df.iloc[[i]]
        forecast_date = test.index[0]

        if forecast_date < forecast_start or forecast_date > forecast_end:
            continue

        if model_kind == "ols":
            model = LinearRegression()
            train_features, test_features = train[predictors], test[predictors]
            if impute_missing:
                controls = [column for column in AR6_PREDICTORS if column in predictors]
                if not np.isfinite(train_features[controls]).all().all() or not np.isfinite(test_features[controls]).all().all():
                    raise ValueError("AR controls must be finite; only additional predictors may be imputed.")
                medians = training_medians(train_features)
                train_features = train_features.fillna(medians)
                test_features = test_features.fillna(medians)
            model.fit(train_features, train[TARGET])
            prediction = model.predict(test_features)[0]
            coefficients = {"intercept": model.intercept_}
            coefficients.update(dict(zip(predictors, model.coef_)))

        elif model_kind == "lasso":
            prediction, coefficients = fit_predict_lasso_with_unpenalized_controls(
                train=train,
                test=test,
                predictors=predictors,
                alphas=lasso_alphas,
                impute_missing=impute_missing,
            )

        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        prediction_rows.append({"date": forecast_date, "prediction": prediction})

        coefficient_row = {"date": forecast_date}
        coefficient_row.update(coefficients)
        coefficient_rows.append(coefficient_row)

    predictions = pd.DataFrame(prediction_rows).set_index("date")["prediction"]
    coefficients = pd.DataFrame(coefficient_rows).set_index("date")
    return predictions, coefficients


def get_or_create_ar6_benchmark(
    df: pd.DataFrame,
    window_size: int = ROLLING_WINDOW_SIZE,
    forecast_start: str | pd.Timestamp = TEST_START,
    forecast_end: str | pd.Timestamp = TEST_END,
    cache_label: str | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    forecast_start = pd.Timestamp(forecast_start)
    forecast_end = pd.Timestamp(forecast_end)

    if cache_label is None and window_size == ROLLING_WINDOW_SIZE:
        benchmark_path = AR6_BENCHMARK_PATH
        coefficient_path = RESULTS_DIR / "paye_monthly_ar6_benchmark_coefficients.csv"
    else:
        label = cache_label or forecast_period_label(forecast_start, forecast_end)
        benchmark_path = (
            RESULTS_DIR
            / f"paye_monthly_ar6_benchmark_predictions_{label}_window_{window_size}.csv"
        )
        coefficient_path = (
            RESULTS_DIR
            / f"paye_monthly_ar6_benchmark_coefficients_{label}_window_{window_size}.csv"
        )

    if benchmark_path.exists() and not force_recompute:
        benchmark = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
        expected_index = df.iloc[window_size:].loc[forecast_start:forecast_end].index
        expected_actual = df.loc[expected_index, TARGET]
        cached_actual = benchmark["actual"].reindex(expected_index) if "actual" in benchmark else None
        coefficient_columns_match = False
        if coefficient_path.exists():
            coefficient_columns = pd.read_csv(coefficient_path, nrows=0).columns
            coefficient_columns_match = set(AR6_PREDICTORS).issubset(coefficient_columns)
        cached_actual_matches = (
            cached_actual is not None
            and cached_actual.notna().all()
            and np.allclose(
                cached_actual,
                expected_actual,
                equal_nan=True,
            )
        )
        if benchmark.index.equals(expected_index) and cached_actual_matches and coefficient_columns_match:
            return benchmark

    predictions, coefficients = rolling_model_predictions(
        df=df,
        predictors=AR6_PREDICTORS,
        model_kind="ols",
        window_size=window_size,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
    )

    benchmark = pd.DataFrame(
        {
            "actual": df.loc[predictions.index, TARGET],
            "benchmark": predictions,
        },
        index=predictions.index,
    )
    benchmark.index.name = "date"

    benchmark.to_csv(benchmark_path)
    coefficients.to_csv(coefficient_path)

    return benchmark


def evaluate_against_benchmark(
    actual: pd.Series,
    benchmark: pd.Series,
    predictions: dict[str, pd.Series],
) -> pd.DataFrame:
    benchmark_mse = mean_squared_error(actual, benchmark)
    rows = [
        {
            "model": "ar6_benchmark",
            "MSE": benchmark_mse,
            "RMSE": np.sqrt(benchmark_mse),
            "OOS_R2": np.nan,
        }
    ]

    for model_name, model_predictions in predictions.items():
        aligned = pd.concat([actual, benchmark, model_predictions], axis=1).dropna()
        y_true = aligned.iloc[:, 0]
        y_benchmark = aligned.iloc[:, 1]
        y_pred = aligned.iloc[:, 2]

        model_mse = mean_squared_error(y_true, y_pred)
        sample_benchmark_mse = mean_squared_error(y_true, y_benchmark)

        rows.append(
            {
                "model": model_name,
                "MSE": model_mse,
                "RMSE": np.sqrt(model_mse),
                "OOS_R2": 1 - model_mse / sample_benchmark_mse,
            }
        )

    return pd.DataFrame(rows)
