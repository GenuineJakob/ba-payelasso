# src/paye_monthly_utils.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target"
TEST_START = "2001-01-01"
TEST_END = "2020-12-31"
ROLLING_WINDOW_SIZE = 360
LASSO_ALPHAS = np.logspace(-5, 1, 200)

AR6_PREDICTORS = [f"vol_lag_{lag}" for lag in range(1, 7)]

PAYE_MONTHLY_PREDICTORS = {
    "cay": "cay",
    "cp": "commercial_paper_spread",
    "dfr": "dfr",
    "dfy": "default_spread",
    "exret": "exret",
    "ik": "ik",
    "ip": "ip_growth",
    "ipvol": "ip_volatility",
    "npy": "npy",
    "ppivol": "ppi_volatility",
    "tms": "term_spread",
}

AR6_BENCHMARK_PATH = RESULTS_DIR / "paye_monthly_ar6_benchmark_predictions.csv"


def load_monthly_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run `python src/build_dataset.py` first."
        )

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()

    for lag in range(1, 7):
        df[f"vol_lag_{lag}"] = df["volatility"].shift(lag - 1)

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


def make_lasso_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lasso",
                LassoCV(
                    alphas=LASSO_ALPHAS,
                    cv=TimeSeriesSplit(n_splits=5),
                    max_iter=50000,
                    random_state=42,
                ),
            ),
        ]
    )


def rolling_model_predictions(
    df: pd.DataFrame,
    predictors: list[str],
    model_kind: str,
) -> tuple[pd.Series, pd.DataFrame]:
    prediction_rows = []
    coefficient_rows = []

    for i in range(ROLLING_WINDOW_SIZE, len(df)):
        train = df.iloc[i - ROLLING_WINDOW_SIZE : i]
        test = df.iloc[[i]]
        forecast_date = test.index[0]

        if forecast_date < pd.Timestamp(TEST_START) or forecast_date > pd.Timestamp(TEST_END):
            continue

        if model_kind == "ols":
            model = LinearRegression()
            model.fit(train[predictors], train[TARGET])
            prediction = model.predict(test[predictors])[0]
            coefficients = {"intercept": model.intercept_}
            coefficients.update(dict(zip(predictors, model.coef_)))

        elif model_kind == "lasso":
            model = make_lasso_model()
            model.fit(train[predictors], train[TARGET])
            prediction = model.predict(test[predictors])[0]
            lasso = model.named_steps["lasso"]
            coefficients = {
                "intercept": lasso.intercept_,
                "alpha": lasso.alpha_,
            }
            coefficients.update(dict(zip(predictors, lasso.coef_)))

        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        prediction_rows.append({"date": forecast_date, "prediction": prediction})

        coefficient_row = {"date": forecast_date}
        coefficient_row.update(coefficients)
        coefficient_rows.append(coefficient_row)

    predictions = pd.DataFrame(prediction_rows).set_index("date")["prediction"]
    coefficients = pd.DataFrame(coefficient_rows).set_index("date")
    return predictions, coefficients


def get_or_create_ar6_benchmark(df: pd.DataFrame, force_recompute: bool = False) -> pd.DataFrame:
    if AR6_BENCHMARK_PATH.exists() and not force_recompute:
        benchmark = pd.read_csv(AR6_BENCHMARK_PATH, index_col=0, parse_dates=True)
        expected_index = df.iloc[ROLLING_WINDOW_SIZE:].loc[TEST_START:TEST_END].index
        if benchmark.index.equals(expected_index):
            return benchmark

    predictions, coefficients = rolling_model_predictions(
        df=df,
        predictors=AR6_PREDICTORS,
        model_kind="ols",
    )

    benchmark = pd.DataFrame(
        {
            "actual": df.loc[predictions.index, TARGET],
            "benchmark": predictions,
        },
        index=predictions.index,
    )
    benchmark.index.name = "date"

    benchmark.to_csv(AR6_BENCHMARK_PATH)
    coefficients.to_csv(RESULTS_DIR / "paye_monthly_ar6_benchmark_coefficients.csv")

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