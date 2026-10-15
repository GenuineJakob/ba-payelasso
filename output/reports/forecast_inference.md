# Forecast uncertainty for the selected thesis comparisons

Generated 2026-10-13T14:36:03+00:00. Existing forecast files are read without refitting or modification.

## Primary comparisons

| Comparison | Period | N | MSE reduction vs reference | 95% block-bootstrap CI | HAC p (two-sided) | Holm p | Assessment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original rolling Lasso vs AR(6) | main | 229 | -0.042% | [-0.182%, +0.015%] | 0.1942 | 0.3884 | Improvement not established (Holm 5%) |
| Expanded macro Lasso vs AR(6) | main | 240 | +1.171% | [+0.016%, +2.570%] | 0.2401 | 0.3884 | Improvement not established (Holm 5%) |
| Expanded macro Lasso vs OLS | main | 240 | +64.686% | [+33.010%, +83.205%] | 0.0714 | 0.2143 | Improvement not established (Holm 5%) |

## Later-period robustness

| Comparison | Period | N | MSE reduction vs reference | 95% block-bootstrap CI | HAC p (two-sided) | Holm p | Assessment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Expanded macro Lasso vs AR(6) | later | 35 | +0.016% | [-0.000%, +0.062%] | 0.2771 | 0.2771 | Improvement not established (Holm 5%) |
| Expanded macro Lasso vs OLS | later | 35 | +86.964% | [+66.074%, +93.132%] | 0.0246 | 0.0491 | Lower Lasso MSE (Holm 5%) |

## Fixed-model diagnostic

| Comparison | Period | N | MSE reduction vs reference | 95% block-bootstrap CI | HAC p (two-sided) | Holm p | Assessment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original fixed Lasso vs AR(6) | main | 240 | +0.000% | [+0.000%, +0.000%] | Unavailable | Unavailable | No valid test: numerically identical losses |

## Method, scope and limitations

Three primary comparisons are selected from already inspected results: original 360-month rolling Lasso vs AR(6), expanded 99-variable macro-only Lasso vs AR(6), and that same expanded Lasso vs corrected OLS. Each pair uses its own identical origin dates, target dates and actuals. Original and expanded designs have different windows and samples; they are not pooled or compared as a common-sample experiment. Fixed Lasso is a diagnostic; the two later-period expanded comparisons are retrospective robustness.

Positive d_t = reference squared error − Lasso squared error favors Lasso. MSE reduction = 1 − mean(Lasso loss)/mean(reference loss). The OLS comparison uses OLS, not AR, in the denominator. Forecast targets are annualized volatility in levels, one month ahead. Percentages describe error reductions, not explained variance.

The test is the two-sided unconditional mean-loss comparison in Giacomini–White (2006), section 3.4, under a fixed estimation-window framework. It uses z = mean(d)/sqrt(HAC long-run variance/n), a standard-normal reference, centered autocovariances with denominator n and Bartlett weights. The default bandwidth is floor(4*(n/100)^(2/9)); lags 0, 3, 6 and 12 are reported as finite-sample sensitivities. This is not the conditional GW instrument test and does not assume loss differences are independent merely because forecasts are one-step ahead. [Method and assumptions](https://economia.uc3m.es/jgonzalo/teaching/PhdTimeSeries/GiacominiWhite.pdf).

Fixed-length rolling fitting and fold-local tuning form the forecasting procedure being compared; the GW framework permits general estimation methods. Validity still requires weak dependence, sufficient moments and nonzero long-run variance. Heavy crisis losses, short evaluation samples and departures from these assumptions can affect the approximation. A nearly identical or constant loss differential produces an unavailable test, not an arbitrary p-value. No Clark–West adjustment or Lasso coefficient significance test is applied.

Paired circular block bootstrap: 20,000 resamples, random seed 20261013, 95% percentile intervals, main block length 6 months; lengths 3, 6 and 12 are shown as sensitivity. Uniform random block starts wrap at the sample end; consecutive blocks are concatenated and trimmed to n. Both losses use the same sampled dates, and each replicate recomputes both MSEs and their ratio. This assumes approximate stationarity and weak dependence of the joint saved losses. [Block-bootstrap background](https://mathweb.ucsd.edu/~politis/impactBOOT.pdf).

The bootstrap resamples saved losses; it does not refit models, retune alpha or repeat the historical model search. The intervals do not capture all uncertainty from model selection and are pointwise, not simultaneous. They are not intervals for future individual volatility observations. The HAC and bootstrap procedures use different approximations, so their decisions need not agree exactly.

Holm adjustment controls multiplicity across the three selected primary HAC tests. The two later-period tests have a separate exploratory family. Sensitivity results are not additional confirmatory tests. This adjustment does not correct the earlier search across models, grids or periods: these are selected, retrospective comparisons, not a prespecified confirmatory exercise. Later results have only 35 observations and are not an untouched holdout.

A confidence interval including zero or a non-rejection does not establish equivalence or prove that macroeconomic information has zero forecasting value. A positive significant difference supports lower average squared error for the compared procedures under the stated assumptions; it identifies neither causation nor significant individual macroeconomic coefficients.

## Sensitivity and reproducibility

The full block-length and HAC-lag sensitivity tables are in [forecast_inference_bootstrap_sensitivity.csv](../tables/forecast_inference_bootstrap_sensitivity.csv) and [forecast_inference_hac_sensitivity.csv](../tables/forecast_inference_hac_sensitivity.csv). Paired losses and dates are exported in [forecast_inference_losses.csv](../results/forecast_inference_losses.csv). All configuration, source checksums and output checksums are in [forecast_inference_manifest.json](../tables/forecast_inference_manifest.json).

Run from Code: `python src/forecast_inference.py`, then `python src/build_project_report.py`. The default seed and configuration reproduce the CSV results; timestamps can change. No training is performed.
