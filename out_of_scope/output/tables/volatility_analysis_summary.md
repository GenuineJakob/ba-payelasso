# Volatility analysis results

Both new experiments were evaluated in log volatility (primary specification)
and volatility levels. Positive percentages below mean lower mean squared
error than the corresponding AR benchmark; negative percentages mean worse
performance. Percentages compare models within the same response scale.

## Same-month explanation: Lasso versus AR(6)

Models are fitted on the preceding 240 months, then evaluated using that
month's predictors observed ex post. There are 240 evaluation months in
2001–2020 and 35 in January 2021–November 2023.

| Scale | Predictor pool | Main: 2001–2020 | Later: 2021–2023 |
|---|---|---:|---:|
| Log | Nine-variable baseline | +0.68% | −1.40% |
| Log | FRED-MD macro (99 predictors) | −40.22% | −0.78% |
| Log | FRED-MD macro/financial, without VIX (124) | +33.56% | +30.55% |
| Log | Full FRED-MD, including VIX (125) | +52.83% | +49.84% |
| Levels | Nine-variable baseline | +3.81% | −3.13% |
| Levels | FRED-MD macro | −79.17% | −2.44% |
| Levels | FRED-MD macro/financial, without VIX | +42.53% | +15.36% |
| Levels | Full FRED-MD, including VIX | +67.23% | +56.72% |

Financial indicators add substantial explanatory value in both periods. The
full log specification selects VIX in every evaluation window. The financial
pool without VIX also improves performance, so the association is not confined
to implied volatility. Same-month S&P price changes are among its available
indicators. These results describe observed market conditions; they do not
establish predictive ability before those conditions are observed or causality.

Macroeconomic variables alone do not give a consistent improvement in this
setup. Joint baseline OLS reduces log-scale error by 1.92% in the main period,
but increases it by 16.88% later. Individual baseline results are also mixed:
commercial-paper and default spreads reduce main-period log error by 4.79%
and 2.25%; inflation reduces later-period error by 4.06% but does not improve
the main period. These comparisons do not establish absence of a macroeconomic
relationship, nor justify choosing a variable solely from its later results.

## Next-quarter forecasting: baseline models versus AR(2)

Quarterly volatility is constructed from squared daily log returns. Models
use 80-quarter rolling windows, previous-quarter macro indicators and
current-quarter financial indicators. The response periods contain 80
quarters in 2001Q1–2020Q4 and 12 in 2021Q1–2023Q4.

| Scale | Model | Main: 2001–2020 | Later: 2021–2023 |
|---|---|---:|---:|
| Log | Lasso | +0.48% | −0.43% |
| Log | Joint OLS | −27.85% | +15.00% |
| Levels | Lasso | −2.59% | −5.85% |
| Levels | Joint OLS | −27.70% | +9.34% |

Changing to quarterly frequency does not produce a consistent forecast gain.
The small main-period log Lasso advantage does not persist later or in levels.
Joint OLS's later advantage follows poor main-period performance. No individual
baseline predictor beats AR(2) in the main period on either scale. The later
period contains only 12 quarterly observations, limiting conclusions about
its improvements.

## Reproducibility and checks

Run instructions and exact data definitions are in
[`README_volatility_analysis.md`](../../README_volatility_analysis.md).
All detailed CSVs have `contemporaneous_` or `quarterly_` prefixes and include
scale and period; prediction files include both origin and response dates.

The test suite passes 50 tests. Full exports were independently checked by
reconstructing predictions and training R-squared from coefficients and
training-window medians, reconciling MSE/RMSE/error-reduction metrics and
checking matching evaluation calendars and responses. AR terms and intercepts
remain unpenalized. No Lasso convergence warnings were recorded. OLS now
standardizes each training window and uses an explicit conventional SVD
cutoff. Corrected joint fits have full numerical rank. The earlier broad-pool
rank shortfalls were due to a scale-sensitive `1e-6` cutoff, not evidence of
mathematical dependence. Detailed rank and conditioning diagnostics are saved.

These are revised-data experiments with assumed reporting delays, not
historical real-time reconstructions. The later period is retrospective
robustness. The existing daily-return definition is retained; this is not an
exact Paye replication. Source data remain unchanged. Before the 2026-10-12
OLS correction, earlier outputs were archived in
`output/archive/ols_before_2026-10-12/`. Only OLS was reestimated; the saved AR
and Lasso predictions were reused unchanged.
