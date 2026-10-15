# Same-month explanation and quarterly forecasts

Run these commands from the `Code` directory, using the Python environment
with `requirements-fredmd.txt` installed. The pinned FRED-MD download and the
existing raw data must already be present.

```bash
python src/build_volatility_analysis_datasets.py
python src/train_contemporaneous.py --period all
python src/train_quarterly.py --period all
```

Both runners default to `--scale both` (log first, then levels) and
`--period main`. Other choices are `--scale log|level|both` and
`--period main|later|all`. For a short check, add `--max-forecasts 2`;
smoke outputs have their own `_smoke` suffix. Custom `--processed-dir`,
`--results-dir` and `--tables-dir` are supported. Rebuilding verifies the
pinned snapshot's source checksums; runners also verify sources and generated
datasets against their frozen build manifests.

## What the experiments measure

**Contemporaneous**: current-month volatility is explained by six previous
months of volatility and reference-month predictors. The model is fitted on
the preceding 240 months. Publication-delay shifts are removed, so held-out
explanatory performance uses current-month indicators observed ex post. It is
not a real-time forecast and does not identify causal effects. Four predictor
pools distinguish the nine-variable baseline, FRED-MD macro indicators,
macro/financial indicators excluding VIX, and the full pool including VIX.
Eligibility is frozen on January 1981–December 2000 at 95% finite coverage.
Single-predictor OLS runs cover each baseline feature. Broad-pool OLS is an
unpenalized comparison with explicit numerical-rank and conditioning diagnostics.
It is not inherently rank deficient: the earlier low reported ranks resulted
from a scale-sensitive solver cutoff.

**Quarterly**: next-quarter volatility is forecast from the current and
previous quarter's volatility (AR(2)) and the nine-variable baseline. Models
use an 80-quarter rolling training window. Macroeconomic predictors are delayed
one quarter; financial indicators use the origin quarter. At the first origin,
2000Q4, training origins are 1980Q4–2000Q3 and their responses are
1981Q1–2000Q4. Those responses have all been observed by the origin.

Quarterly annualized volatility is `sqrt(4 * sum(daily log return squared))`.
Returns are calculated before grouping, so boundary returns belong to their
ending date. The initial quarter lacks an opening return and is excluded;
quarters missing an entire calendar month or ending clearly early are marked
unavailable. No exchange-calendar reconstruction or correction of unobserved
daily prices is attempted. The cached S&P source ends December 29, 2023.
Quarterly volatility is not the mean of monthly volatilities.

Quarterly indices and rate/spread indicators require three finite monthly
observations before taking their mean. CPI/IP growth is the four-quarter log
change of quarterly mean indices. IP/PPI growth volatility is the sample
standard deviation (`ddof=1`) of the latest four annual-growth observations.
Modern commercial-paper data take precedence over the historical series.
An incomplete monthly block becomes a missing predictor. There is no future
backfill or interpolation. Eligibility is frozen on the initial 80-quarter
training window at 95% finite coverage.

In both analyses, log specifications apply the logarithm to the response and
all AR terms, never to unrelated predictors. Nonpositive observed volatility
is rejected; no arbitrary log offset is introduced. AR coefficients and the
intercept remain unpenalized. Only additional predictors enter Lasso's L1
penalty. The existing 200-alpha grid and five chronological folds are retained.
Median imputation, AR residualization and scaling are estimated separately
within each fold. OLS medians use only its training window. All-missing or
constant additional columns have zero coefficients.

OLS standardizes all fitted columns on the training window and uses
`numpy.linalg.lstsq` with `rcond = eps(float64) * max(n_train, n_predictors)`.
The intercept and coefficients are exported in original units. Diagnostics
record the cutoff, standardized condition number and raw-design ranks at
conventional and `1e-6` cutoffs. Run manifests record package versions.
`python src/rerun_ols.py` refreshes only saved OLS predictions, coefficients,
metrics and diagnostics, archives their prior versions, and preserves AR/Lasso.

## Dates, outputs and interpretation

Monthly evaluation months are January 2001–December 2020 (`main`) and January
2021–November 2023 (`later`). Quarterly target periods are 2001Q1–2020Q4 and
2021Q1–2023Q4; quarterly CSV `date` is the forecast origin and `target_date` is
the following quarter-end. The later period is retrospective robustness,
not a fresh untouched holdout.

New datasets and build manifests go in `data/processed`, predictions,
long-format coefficients, missing-value medians and run manifests in
`output/results`, and model comparisons, selection frequencies, data-quality
tables and window diagnostics in `output/tables`. Names start with
`contemporaneous_` or `quarterly_` and include scale and period. Existing
monthly forecast files remain unchanged.

Metrics are calculated in each model's native response scale. Compare each
model with its matching-scale AR benchmark using `error_reduction_vs_AR`:
positive means lower squared error. For the monthly analysis this is exported
as `heldout_explanatory_R2_vs_AR`; for quarterly forecasting it is `OOS_R2`.
These are relative to AR error, not conventional R-squared against the mean.
`mean_training_R2` is conventional training R-squared averaged over windows.
There is no exponentiation or level-scale bias adjustment of log predictions;
raw log/level MSEs should not be ranked against each other. Coefficients are in
original predictor units, with the response in the stated scale. Selection
frequencies count only additional Lasso predictors. Diagnostics flag OLS rank
deficiency, imputation rates, convergence warnings and Lasso dual gaps.

These are revised-data experiments with assumed delays. They follow Paye's
AR(6)/AR(2) lag orders and log transformation, but use the existing S&P 500
log index returns rather than reproducing his excess-return construction.
Better contemporaneous explanation does not establish better forecasts.
