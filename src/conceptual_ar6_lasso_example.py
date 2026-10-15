"""A simplified, readable sketch of the AR(6) benchmark and LASSO exercise.

This file is for explaining the empirical design, not for producing the final
results.  The production implementation is in ``paye_monthly_utils.py`` and
the ``train_paye_monthly_*.py`` files.

The key distinction is:

1. The AR(6) benchmark is fitted as its own OLS model.
2. The LASSO uses the six AR variables as unpenalized controls.
3. Only the additional predictor is subject to the L1 penalty.
4. The benchmark's *prediction* is not an input to the LASSO.
5. The two sets of predictions are compared only after estimation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from partially_penalized_lasso import PartiallyPenalizedLassoCV


# ---------------------------------------------------------------------------
# 1. Load data and create the autoregressive variables
# ---------------------------------------------------------------------------

data_file = Path(__file__).resolve().parents[1] / "data" / "processed" / "dataset.csv"
data = pd.read_csv(data_file, index_col=0, parse_dates=True).sort_index()

# Current volatility and its five lags form the six-variable AR block used in
# the main code. The target variable is the quantity being forecast.
data["vol"] = data["volatility"]
data["vol_lag_1"] = data["volatility"].shift(1)
data["vol_lag_2"] = data["volatility"].shift(2)
data["vol_lag_3"] = data["volatility"].shift(3)
data["vol_lag_4"] = data["volatility"].shift(4)
data["vol_lag_5"] = data["volatility"].shift(5)

ar_variables = [
    "vol",
    "vol_lag_1",
    "vol_lag_2",
    "vol_lag_3",
    "vol_lag_4",
    "vol_lag_5",
]

# Use one PAYE variable here so the model structure is easy to see. In the
# actual analysis, this step is repeated for other variables or blocks.
additional_variable = "commercial_paper_spread"

data = data.dropna(subset=["target"] + ar_variables + [additional_variable])


# ---------------------------------------------------------------------------
# 2. Produce rolling one-period-ahead forecasts
# ---------------------------------------------------------------------------

window_size = 240
forecast_start = pd.Timestamp("2001-01-01")
forecast_end = pd.Timestamp("2020-12-31")

forecast_dates = []
actual_values = []
ar6_predictions = []
lasso_predictions = []

for row_number in range(window_size, len(data)):
    training_data = data.iloc[row_number - window_size : row_number]
    forecast_row = data.iloc[[row_number]]
    forecast_date = forecast_row.index[0]

    if forecast_date < forecast_start or forecast_date > forecast_end:
        continue

    # MODEL A: standalone AR(6)-style OLS benchmark
    ar6_model = LinearRegression()
    ar6_model.fit(training_data[ar_variables], training_data["target"])
    ar6_forecast = ar6_model.predict(forecast_row[ar_variables])[0]

    # MODEL B: AR variables are unpenalized controls; only the additional
    # predictor is penalized. `ar6_forecast` itself is not a LASSO input.
    # The shared estimator removes the AR contribution from both the target
    # and the additional predictor, then applies Lasso to the residuals.
    # Projections and scaling are fitted separately inside each CV training
    # fold. Finally, OLS recovers the unpenalized intercept and AR coefficients.
    predictors = ar_variables + [additional_variable]
    lasso_model = PartiallyPenalizedLassoCV(
        controls=ar_variables,
        alphas=np.logspace(-5, 1, 200),
    )
    lasso_model.fit(training_data[predictors], training_data["target"])
    lasso_forecast = lasso_model.predict(forecast_row[predictors])[0]

    forecast_dates.append(forecast_date)
    actual_values.append(forecast_row["target"].iloc[0])
    ar6_predictions.append(ar6_forecast)
    lasso_predictions.append(lasso_forecast)


# ---------------------------------------------------------------------------
# 3. Compare forecasts after both models have been estimated
# ---------------------------------------------------------------------------

results = pd.DataFrame(
    {
        "actual": actual_values,
        "AR6_benchmark": ar6_predictions,
        "LASSO": lasso_predictions,
    },
    index=forecast_dates,
)

ar6_mse = mean_squared_error(results["actual"], results["AR6_benchmark"])
lasso_mse = mean_squared_error(results["actual"], results["LASSO"])
lasso_oos_r_squared = 1 - lasso_mse / ar6_mse

print("AR(6) benchmark MSE:", ar6_mse)
print("LASSO MSE:", lasso_mse)
print("LASSO out-of-sample R-squared relative to AR(6):", lasso_oos_r_squared)

# Positive OOS R-squared: LASSO forecasts better than the AR(6) benchmark.
# Negative OOS R-squared: AR(6) benchmark forecasts better than the LASSO.
