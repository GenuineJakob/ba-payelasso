# Uncertainty for the selected thesis forecasts

Run from `Code` in the existing project environment:

```bash
python src/forecast_inference.py
python src/build_project_report.py
python -m pytest tests -q -p no:cacheprovider
```

Only saved forecasts are evaluated. No model is retrained, no prediction is
clipped, no crisis observation is removed, and no existing model output changes.
`--resamples` (default 20,000; minimum 1,000) and `--seed` (default 20261013)
control the bootstrap. `--root` permits evaluation of a separate project copy.
NumPy, pandas and SciPy are required; SciPy is already a scikit-learn dependency.

## Scope

Three selected primary comparisons:

1. Original ten-variable **360-month rolling** Lasso vs AR(6).
2. Expanded **99-variable macro-only, 240-month rolling** Lasso vs AR(6).
3. The same expanded Lasso vs corrected, training-standardized OLS.

The two expanded comparisons are repeated for the 35-observation later period
as retrospective robustness. Fixed Lasso vs AR is an additional diagnostic;
identical forecasts have no meaningful comparison-test variance. Paye blocks,
financial pools, same-month explanation and quarterly forecasts are not tested.
These comparisons were selected after inspecting results, not prespecified.
The original and expanded designs have different samples and windows: their
loss sequences are not pooled into a common-sample comparison.

Dates must match exactly within each comparison, be consecutive calendar month
ends, and predict the following month. Actual targets and benchmark values must
agree. Source-table MSEs are independently reconciled before inference; no
inner join drops difficult observations to make pairing work. FRED-MD run
manifests and window diagnostics must describe full 240-month, five-fold runs.
Older fixed exports have an unnamed date-index column and no explicit target
date; the next-month target date is derived according to that saved design.

## Tests and intervals

`d_t = reference squared error - Lasso squared error`. A positive average
favors Lasso. The percentage reduction is
`100 * (1 - MSE_Lasso / MSE_reference)`; reference means **OLS** for the OLS
comparison, not AR.

The two-sided **unconditional** mean-loss comparison follows section 3.4 of
[Giacomini and White (2006)](https://economia.uc3m.es/jgonzalo/teaching/PhdTimeSeries/GiacominiWhite.pdf).
It uses a standard-normal statistic and a centered Bartlett/Newey–West
long-run variance estimate with denominator n and no finite-sample correction.
The default lag bandwidth is `floor(4*(n/100)^(2/9))`, a deterministic heuristic
(4 for the two main samples, 3 for the later sample); 0, 3, 6 and 12 lags are
sensitivity settings. One-step forecasting does not imply independent losses.
This is not a conditional GW test with state-dependent instruments, a
Clark–West adjusted test, or a significance test for selected coefficients.

The fixed-window framework treats repeated fitting and fold-local alpha
tuning as part of the forecasting procedure. It still needs weak dependence,
sufficient moments and a nonzero long-run loss-difference variance. An
identical or constant loss difference produces an unavailable p-value rather
than a fabricated one. Normal asymptotics can be unreliable for short or
crisis-dominated samples; bandwidth sensitivity is therefore reported.

Paired **circular block-bootstrap percentile** intervals resample consecutive
blocks of the two loss sequences together. Blocks have uniformly sampled
start dates, wrap at the sample end, and are concatenated and trimmed to the
original sample length. The main block length is six months; three and twelve
months are sensitivity settings. Each replicate recomputes both MSEs and the
ratio. Configuration and the individual deterministic seed for each
comparison/block are exported. See [Politis's author review of block
resampling](https://mathweb.ucsd.edu/~politis/impactBOOT.pdf).

These are 95% pointwise intervals, assuming approximate stationarity and weak
dependence of the joint saved losses. They do not refit models, repeat tuning,
account for the historical specification search or estimate future individual
volatility prediction intervals. A zero reference MSE in any bootstrap
replicate makes the ratio interval unavailable instead of silently dropping
those replicates.

Holm adjustment applies to the three selected main HAC p-values. Later tests
form a separate exploratory family of two. Undefined tests count in the
family size but have no adjusted p-value. Confidence intervals and sensitivity
tests are not multiplicity-adjusted. Adjustment among these comparisons does
not undo the earlier search across models, grids and periods.

**Report both tests and intervals.** Percentile ratio intervals can disagree
with asymptotic mean-loss tests, particularly for skewed crisis-driven losses;
do not pick the favorable result and call it definitive. Non-rejection is not
proof of equivalence or absence of predictive information. Neither procedure
identifies causation or significant individual macroeconomic coefficients.

## Artifacts and provenance

- `output/tables/forecast_inference_summary.csv`: point estimates, HAC
  standard errors, mean-loss confidence intervals, raw/adjusted p-values,
  bootstrap ratio/difference confidence intervals, status and dates.
- `output/tables/forecast_inference_hac_sensitivity.csv`: all lag settings.
- `output/tables/forecast_inference_bootstrap_sensitivity.csv`: all block sizes.
- `output/results/forecast_inference_losses.csv`: paired forecasts, squared
  errors and differences for all comparisons, in date order.
- `output/tables/forecast_inference_manifest.json`: assumptions, configuration,
  software versions, input/output hashes and unchanged prediction-file hashes.
- `output/reports/forecast_inference.md`: concise standalone methods/results.
- Existing Markdown reports and `results.html`: integrated section 5a.

CSV reductions and interval endpoints are **fractions**; multiply by 100 for
percent. Loss differences and their HAC intervals have squared-volatility
units. Blank p-values are unavailable tests, not non-significant p-values.
The report builder verifies the inference source/output hashes and refuses
stale results if a source forecast or metric table changes. Rerun inference
after rerunning forecasts or correcting metrics. Prior reports and the report
builder are archived before this addition; training scripts and model outputs
are preserved.
