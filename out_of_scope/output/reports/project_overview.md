# Project overview: stock-market volatility, macroeconomics and Lasso

Report generated 2026-10-13T14:36:37+00:00. This overview describes the current code and saved experiments. The report builder reads existing outputs; it does not refit models or alter input data. The companion [results report](results_report.md) and [browsable page](results.html) contain the numerical comparisons.

## 1. Research question and what has been learned

The project asks whether macroeconomic and financial information improves predictions of S&P 500 realized volatility beyond the persistence already captured by past volatility. Its central comparison is an autoregressive benchmark against the same autoregressive block augmented with OLS or Lasso predictors. The additional experiments distinguish prediction of future volatility from explanation of volatility in the same period, and monthly from quarterly frequency.

The saved experiments show small and unstable incremental monthly forecast gains from additional predictors. Broader FRED-MD macro pools do not produce a consistent advantage in the later period. Same-month financial indicators, particularly when implied volatility is included, explain current volatility much better than the AR benchmark. Quarterly forecasting does not deliver a consistent improvement. These conclusions apply to the specific samples, transformations, delays and linear models tested; they do not prove that macroeconomic conditions are unrelated to volatility.

Paye's research motivates the incremental-information question and the lag orders: AR(6) for monthly data and AR(2) for quarterly data. His main target is log annualized volatility, with a different return construction and historical sample. The older project models retain levels; the new contemporaneous and quarterly experiments evaluate both logs and levels. The experiment named `paye_replication` is therefore a Paye-inspired comparison, not an exact replication. [Paye paper](https://vtechworks.lib.vt.edu/bitstream/handle/10919/78015/Dejavol_prepub.pdf?sequence=1).

## 2. Inputs and data coverage

| Input | Coverage | Role |
| --- | --- | --- |
| S&P 500 daily prices | 13,619 daily rows; 1970-01-02 to 2023-12-29 | Daily log index returns and monthly/quarterly realized volatility |
| Original FRED macro cache | 648 monthly dates; 1970–2023; 12 source series | Rates, spreads, prices, production, unemployment, quarterly GDP |
| Processed original monthly dataset | 635 consecutive month ends; 1971-01 to 2023-11 | Volatility, next-month target, 14 candidate explanatory columns |
| Goyal–Welch annual data | Optional cached annual source | Default-return spread, consumption/wealth ratio, investment/capital ratio |
| Cached payout dataset | Monthly payout observations; processed coverage ends in 2019 | Net payout yield in the older Paye models |
| FRED-MD September 2026 vintage | 126 series; 812 observations starting January 1959 | Broad monthly macroeconomic and financial information |

`download_data.py` currently uses tidyfinance to obtain the S&P index and FRED observations, then writes the established CSV formats. External annual and payout sources are read from local caches by the dataset builder. `download_fredmd.py` downloads a pinned September 2026 snapshot and official variable appendix, requires no API key for those files, and checks cached hashes. It does not silently advance to a newer vintage. [Official FRED-MD download page](https://www.stlouisfed.org/research/economists/mccracken/fred-databases).

The FRED-MD snapshot extends beyond the modeling sample. Its later observations are not used to extend the S&P target beyond 2023. Source files, processed datasets and build manifests remain separate. The reporting manifest lists hashes of every file used to produce this report.

The separate options-only robustness caches Cboe daily VIX closes from 01/02/1990 through 10/12/2026, retrieved 2026-10-13T13:48:46.924692+00:00. Only observations through November 2023 enter the matched analysis; the cache is checksum-verified and not silently refreshed.

## 3. Volatility and target construction

```text
Daily return: r_d = log(P_d) − log(P_(d−1))
Monthly annualized volatility: Vol_m = sqrt(12 × Σ_(d in m) r_d²)
Quarterly annualized volatility: Vol_q = sqrt(4 × Σ_(d in q) r_d²)
Log specification: y_t = log(Vol_t); level specification: y_t = Vol_t
```

The code uses the cached `Close` price field, falling back to `Adj Close` where supported. The current downloader maps its adjusted close to `Close`. This is the existing S&P index-return construction, not Paye's daily excess-return construction. A volatility value of 0.20 represents annualized volatility of 20%. Realized volatility here is the square root of a sum of squared daily returns, with annualization; it is not the sample standard deviation of monthly returns.

Returns are computed before aggregation, so a return crossing a month or quarter boundary is assigned to its ending date. Quarterly volatility is built directly from daily returns, not by averaging monthly volatilities. For complete matching quarters, `Vol_q² = (Vol_m1² + Vol_m2² + Vol_m3²)/3`. The quarterly builder excludes the first quarter because its opening return is unavailable; absent calendar months and clearly early terminal quarters are marked unavailable. It does not reconstruct the exchange's entire trading calendar.

The original and FRED-MD forecast datasets index rows by origin month t and put `Vol_(t+1)` in `target`. The contemporaneous dataset instead puts `Vol_t` in `target`, with controls `Vol_(t−1)` through `Vol_(t−6)`. The quarterly dataset indexes the origin q and predicts `Vol_(q+1)` from `Vol_q` and `Vol_(q−1)`. New prediction outputs explicitly store `target_date`. Log runs transform the response and every AR term consistently and reject nonpositive observed volatility instead of adding an offset.

## 4. Predictors and timing

| Nine-variable baseline | Construction | FRED-MD monthly forecast delay | Quarterly forecast availability |
| --- | --- | --- | --- |
| Unemployment | UNRATE level | Two months | Previous quarter |
| Federal funds rate | FEDFUNDS level | Current month | Current quarter |
| Default yield spread | BAA − AAA | Current month | Current quarter |
| Term spread | GS10 − TB3MS | Current month | Current quarter |
| Commercial-paper spread | Modern CP series, historical fallback, minus TB3MS | Current month | Current quarter |
| Inflation | 12-month CPI log change; quarterly: four-quarter log change of quarterly mean CPI | Two months | Previous quarter |
| IP growth | 12-month IP log change; quarterly: four-quarter log change of quarterly mean IP | Two months | Previous quarter |
| IP volatility | 12-month SD of annual IP growth; quarterly: four-quarter SD of annual growth | Two months | Previous quarter |
| PPI volatility | 12-month SD of annual PPI inflation; quarterly: four-quarter SD of annual inflation | Two months | Previous quarter |

Quarterly index, rate and spread means require three finite monthly observations. An incomplete block becomes missing and is later imputed using the training window only. Growth transformations are applied to quarterly mean indices; volatility measures use the latest four annual-growth observations and the sample standard deviation with `ddof=1`. Modern commercial-paper data take precedence, with historical values used where modern observations are missing.

The older Paye menu has eight available predictors: cp, dfr, dfy, ip, ipvol, npy, ppivol and tms. `exret` is absent and is not replaced with zero. The processed dataset also contains GDP, cay and ik, but these do not all enter the Paye menu. GDP is quarterly carried into monthly rows; dfr, cay and ik come from annual data carried forward. These are not new monthly observations. The new baseline excludes those repeated low-frequency inputs and excludes npy because its coverage ends in 2019. The original ten-predictor models retain GDP as part of their historical specification.

| Pool | Eligible extra predictors | Content |
| --- | --- | --- |
| Monthly baseline | 9 | The nine variables listed above |
| FRED-MD macro | 99 | Official groups 1–5 and 7: activity, labor, housing, consumption/orders, money/credit and prices |
| FRED-MD without extended volatility proxy | 124 | Macro plus rates, exchange rates and stock indicators; excludes VIXCLSx |
| Full FRED-MD | 125 | Identical broad financial pool plus the historically spliced VIXCLSx proxy |

FRED-MD transformations come from the pinned CSV's transformation row: level, first difference, second difference, log level, first log difference, second log difference, or change in fractional growth. They use only the current and earlier observations. Invalid logs and divisions become missing, with no future backfill or interpolation. [Transformation documentation](https://files.stlouisfed.org/files/htdocs/wp/2015/2015-012.pdf).

The complete variable names, descriptions, transformations, eligibility decisions and delays are available in the [monthly forecast feature dictionary](../../data/processed/fredmd_feature_metadata.csv), [same-month feature dictionary](../../data/processed/contemporaneous_feature_metadata.csv), and [quarterly feature dictionary](../../data/processed/quarterly_feature_metadata.csv).

Historical VIX correction: `VIXCLSx` is an extended volatility proxy, not observed implied volatility throughout. FRED-MD's original VXO splice includes Bloom's normalized realized-return volatility before 1986; the change log later records replacement of VXOCLSx by VIXCLSx. The cached series starts in July 1962. In this project's 1981–1985 observations, its correlation with realized volatility is 0.99171. This affects early training interpretation; evaluation targets begin in 2001 and are not directly copied into the predictor. The saved dictionaries retain their original source label `VIX`; [the separate provenance audit](../tables/vix_history_audit.json) supplies this correction. [FRED-MD change log](https://files.stlouisfed.org/files/htdocs/econ/mccracken/fred-databases/fredmdchanges.pdf), [Bloom's author draft, Figure 1](https://users.nber.org/~confer/2006/si2006/efg/bloom.pdf).

The new matched robustness uses Cboe's options-based VIX daily closing history from January 1990, averaged within each reference month without backfill. It has no realized-volatility extension. Early modern-methodology VIX observations are back-calculated from options; they are not a historical real-time vintage. The 240-month comparison begins January 2010; a separate 120-month comparison retains January 2001. Every design refits its own AR(6), broad pool without the volatility proxy, original extended proxy, options-only replacement, and AR plus options-only VIX. See the detailed results and [Cboe historical data](https://www.cboe.com/tradable_products/vix/vix_historical_data/).

For forecasting, broad FRED-MD macro indicators have a two-month assumed delay; rates, exchange rates, S&P prices and the extended volatility proxy use the origin month, while S&P dividend yield and P/E have a six-month delay. The same-month explanation removes these shifts and uses reference-month observations ex post. Original/Paye forecasts do not implement this explicit publication-delay schedule, so those saved outputs should be interpreted as revised-data comparisons rather than historical real-time forecasts.

## 5. Models and why the AR block is unpenalized

```text
Monthly forecast: y_(t+1) = a + Σ_(k=0 to 5) φ_(k+1)y_(t−k) + β′X_t + ε_(t+1)
Same-month explanation: y_t = a + Σ_(k=1 to 6) φ_k y_(t−k) + β′X_t + ε_t
Quarterly forecast: y_(q+1) = a + φ_1 y_q + φ_2 y_(q−1) + β′X_q^available + ε_(q+1)
```

- AR benchmark: estimate the intercept and volatility lags by ordinary least squares, with no additional predictors.
- AR + OLS: estimate the AR block and additional predictors jointly without regularization. Adding variables can improve training fit while worsening evaluation performance.
- AR + Lasso: preserve the intercept and complete AR block; penalize only additional predictors. A sufficiently large penalty produces the corresponding AR benchmark.

The shared `PartiallyPenalizedLassoCV` estimator projects both the response and extra predictors on the AR controls, standardizes the remaining predictor residuals, estimates the Lasso coefficients, then recovers the intercept and AR coefficients by conditional OLS. Coefficients are returned in the original predictor units and in the declared response scale. A zero extra coefficient means no incremental contribution in that fitted model, not that the variable has no economic relationship with volatility.

Alpha is selected using five chronological validation folds and complete AR-plus-Lasso validation predictions. Older/general experiments use 200 alphas from 10⁻⁵ to 10; the batch compares 200-value grids from 10⁻⁴ to 1 and from 10⁻³ to 1. Older runs use coordinate descent. FRED-MD and the new analyses use a checked LARS path evaluated on the same requested grid, with numerical polishing if necessary. Internal target normalization is reversed consistently, preserving the penalized objective. Solver diagnostics record relative dual gaps and convergence warnings.

In the new experiments, remaining predictor gaps are filled with training-window medians. Inside every validation fold, imputation, AR residualization and scaling are fitted only on that fold's training observations. All-missing or constant extra predictors have zero coefficients. AR controls and responses must be finite and are never imputed. Feature eligibility is decided once from at least 95% finite observations in the initial 240-month or 80-quarter window, not from later coverage or correlations with evaluation targets. `ACOGNO`, which starts in 1992, fails that initial coverage check.

The older original and Paye workflows instead use common complete-case samples. Requiring every available Paye variable makes even one-variable comparisons end in December 2019 because of npy. This keeps those models comparable within that experiment but sacrifices later observations. Dropping dates could also turn an observation-count rolling window into something other than consecutive calendar months; the new workflows explicitly preserve calendar rows.

## 6. Experiment designs and actual evaluation dates

| Family | Period | Scale | Training design | N per model | First origin | Last origin | First response | Last response |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original fixed | Saved evaluation | level | Fixed fit through 2000 | 240 | 2001-01-31 | 2020-12-31 | 2001-02-28 | 2021-01-31 |
| Original rolling | Saved evaluation | level | 360 months | 229 | 2001-12-31 | 2020-12-31 | 2002-01-31 | 2021-01-31 |
| Paye individual | 2001–2019 | level | 240 months | 228 | 2001-01-31 | 2019-12-31 | 2001-02-28 | 2020-01-31 |
| Paye joint | 2001–2019 | level | 240 months | 228 | 2001-01-31 | 2019-12-31 | 2001-02-28 | 2020-01-31 |
| Paye batch | 2001–2019 | level | 240 months | 228 | 2001-01-31 | 2019-12-31 | 2001-02-28 | 2020-01-31 |
| Paye batch | 1993–2010 | level | 240 months | 216 | 1993-01-31 | 2010-12-31 | 1993-02-28 | 2011-01-31 |
| FRED-MD forecast | Main | level | 240 months | 240 | 2001-01-31 | 2020-12-31 | 2001-02-28 | 2021-01-31 |
| FRED-MD forecast | Later | level | 240 months | 35 | 2021-01-31 | 2023-11-30 | 2021-02-28 | 2023-12-31 |
| Same-month explanation | Main | log | 240 months | 240 | 2001-01-31 | 2020-12-31 | 2001-01-31 | 2020-12-31 |
| Same-month explanation | Later | log | 240 months | 35 | 2021-01-31 | 2023-11-30 | 2021-01-31 | 2023-11-30 |
| Same-month explanation | Main | level | 240 months | 240 | 2001-01-31 | 2020-12-31 | 2001-01-31 | 2020-12-31 |
| Same-month explanation | Later | level | 240 months | 35 | 2021-01-31 | 2023-11-30 | 2021-01-31 | 2023-11-30 |
| Quarterly forecast | Main | log | 80 quarters | 80 | 2000-12-31 | 2020-09-30 | 2001-03-31 | 2020-12-31 |
| Quarterly forecast | Later | log | 80 quarters | 12 | 2020-12-31 | 2023-09-30 | 2021-03-31 | 2023-12-31 |
| Quarterly forecast | Main | level | 80 quarters | 80 | 2000-12-31 | 2020-09-30 | 2001-03-31 | 2020-12-31 |
| Quarterly forecast | Later | level | 80 quarters | 12 | 2020-12-31 | 2023-09-30 | 2021-03-31 | 2023-12-31 |
| Options-only VIX robustness | Main | log | 120 months | 240 | 2001-01-31 | 2020-12-31 | 2001-01-31 | 2020-12-31 |
| Options-only VIX robustness | Later | log | 120 months | 35 | 2021-01-31 | 2023-11-30 | 2021-01-31 | 2023-11-30 |
| Options-only VIX robustness | Main | level | 120 months | 240 | 2001-01-31 | 2020-12-31 | 2001-01-31 | 2020-12-31 |
| Options-only VIX robustness | Later | level | 120 months | 35 | 2021-01-31 | 2023-11-30 | 2021-01-31 | 2023-11-30 |
| Options-only VIX robustness | Main | log | 240 months | 132 | 2010-01-31 | 2020-12-31 | 2010-01-31 | 2020-12-31 |
| Options-only VIX robustness | Later | log | 240 months | 35 | 2021-01-31 | 2023-11-30 | 2021-01-31 | 2023-11-30 |
| Options-only VIX robustness | Main | level | 240 months | 132 | 2010-01-31 | 2020-12-31 | 2010-01-31 | 2020-12-31 |
| Options-only VIX robustness | Later | level | 240 months | 35 | 2021-01-31 | 2023-11-30 | 2021-01-31 | 2023-11-30 |

The original fixed model is trained once through December 2000 and supplies 240 evaluation origins. Its rolling counterpart needs 360 complete monthly rows, so its saved evaluation begins in December 2001 and contains 229 origins. Paye individual and joint runs use 240-month windows with 228 origins in 2001–2019. The batch uses seven predictor blocks and two alpha grids in each of two historical periods; its best row is a retrospective selection across tested configurations.

FRED-MD and contemporaneous analyses use 240 main and 35 later monthly observations. Quarterly analyses use 80 main and 12 later response quarters. Forecast origin dates and response dates differ: the FRED-MD main origins January 2001–December 2020 predict February 2001–January 2021; the same-month analysis explains January 2001–December 2020 itself. Quarterly main origins 2000Q4–2020Q3 predict 2001Q1–2020Q4. These differences matter when interpreting crises and comparing periods.

The filenames call the FRED-MD later period `holdout`; subsequent analysis has already inspected that period. It is now reported here as later-period robustness, not a fresh untouched test. Rolling re-estimation remains chronological, but later model choices are informed by observed results. Small quarterly later-period improvements rest on only 12 observations.

## 7. Evaluation, reproducibility and limits

```text
MSE = mean[(actual − prediction)²]
RMSE = sqrt(MSE)
Error reduction vs AR = 1 − MSE_model / MSE_AR
Displayed percent = 100 × error reduction vs AR
```

Positive error reduction means better accuracy than the matching AR model. For example, 0.0042 in a CSV means about 0.42% lower MSE, not 42%. This relative measure is not conventional R-squared against a historical mean and is not the fraction of total volatility explained. In levels, an RMSE of 0.0897 means about 8.97 annualized volatility percentage points. Log and level MSEs have different units and must not be ranked directly. No exponentiation or mean-bias adjustment is applied to log predictions.

The saved model comparisons describe realized sample performance. Selected monthly forecast comparisons now have dependence-adjusted mean-loss tests and paired block-bootstrap confidence intervals, documented in section 5a of the results report and the separate forecast-inference report. Other model comparisons remain descriptive. No causal identification is attempted. OLS now centers and standardizes predictors using only each training window and solves with an explicit relative singular-value cutoff, `eps(float64) * max(n_train, n_predictors)`. Coefficients are restored to original units. A numerical-rank report depends on this cutoff and scaling; it does not itself prove mathematical dependence. The corrected broad fits have full numerical rank, while their forecast errors remain large.

The earlier implementation used scikit-learn's dense OLS default, which in version 1.9 uses a `1e-6` relative singular-value cutoff. On unscaled data this discarded independent small directions. In the first full FRED-MD window, raw/conventionally scaled numerical rank is 131, while the old cutoff reported 125. The corrected standardized condition number is about 1,081, compared with about 10.3 million for the centered raw design. This is a numerical estimation issue, not evidence that broad OLS is inherently unidentified. [Scikit-learn tolerance documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

The fixed FRED-MD vintage and original cached macro data are revised data. Assumed reporting delays do not undo later revisions or reconstruct historical information sets. The volatility source differs from Paye's excess-return construction. A high contemporaneous selection rate for the extended VIX proxy does not establish an implied-versus-realized relationship over its synthetic early history, or better next-month forecasts. The options-only comparisons address that provenance limitation with matching windows and benchmarks.

The volatility-analysis validation records 50 passing tests and independent reconstruction of 13,758 same-month/quarterly model predictions, including training R-squared and metrics. The OLS correction additionally checked 17,058 prediction rows across FRED-MD and those analyses, retaining AR/Lasso unchanged while reestimating 10,090 full-evaluation OLS fits. The report recalculates metrics for all 278 canonical comparison rows. These numerical checks are separate from testing statistical significance.

## 8. Files, entry points and how to reproduce

| Stage | Scripts under src/ | Purpose |
| --- | --- | --- |
| Raw input acquisition | download_data.py; download_fredmd.py | Existing price/macro CSVs; pinned FRED-MD CSV, appendix and checksum manifest |
| Dataset construction | build_dataset.py; build_fredmd_dataset.py; build_volatility_analysis_datasets.py | Original, delayed FRED-MD, same-month and quarterly datasets |
| Original models | train_models.py | Fixed and 360-month rolling benchmark/OLS/Lasso |
| Paye models | train_paye_monthly_individual.py; train_paye_monthly_kitchen_sink.py; train_paye_monthly_batch_experiments.py | Individual, joint and historical block/grid comparisons |
| Expanded models | train_fredmd.py | Four delayed monthly predictor pools |
| New questions | train_contemporaneous.py; train_quarterly.py | Same-month explanation and next-quarter forecasts in both scales |
| VIX provenance robustness | download_vix.py; train_vix_robustness.py; evaluate_vix_robustness.py | Options-only matched 120/240-month explanatory comparisons and independent verification |
| Shared estimation | partially_penalized_lasso.py; paye_monthly_utils.py; volatility_analysis_utils.py | Unpenalized controls, tuning, imputation, exports and metrics |
| Forecast uncertainty | forecast_inference.py | Selected rolling forecast tests, paired block-bootstrap intervals and sensitivity without refitting |
| Reporting | evaluate_paye_monthly_outputs.py; build_project_report.py | Legacy output summaries and this comprehensive local report |

Run commands from the `Code` directory in an environment with the project dependencies. The FRED-MD requirements cover its models and tests; the reporting environment additionally needs matplotlib. Original downloading additionally needs tidyfinance. Re-running training replaces that experiment's own output files; re-running only this report reads them without training. Preserve the pinned caches if you want the same source snapshot.

```text
# Original comparisons
python src/train_models.py
# Paye-inspired individual, joint and block experiments
python src/train_paye_monthly_individual.py
python src/train_paye_monthly_kitchen_sink.py
python src/train_paye_monthly_batch_experiments.py
# Expanded monthly forecasting
python src/build_fredmd_dataset.py
python src/train_fredmd.py --period all
# Same-month explanation and quarterly forecasting
python src/build_volatility_analysis_datasets.py
python src/train_contemporaneous.py --period all
python src/train_quarterly.py --period all
# Refresh this report without rerunning models
python src/build_project_report.py
```

For the new runners, `--max-forecasts 2` produces separate smoke outputs; these are excluded from this report. Logs and levels are selected with `--scale log|level|both`; new experiment periods use `main|later|all`, while the older FRED-MD runner retains `main|holdout|all`. Dataset-building source checksums guard against silently changing the frozen inputs.

```text
# Cache and evaluate options-only VIX robustness
python src/download_vix.py
python src/train_vix_robustness.py --period all --scale both --window both
python src/evaluate_vix_robustness.py
python src/build_project_report.py
```

For definitions and detailed runner options, see [FRED-MD instructions](../../README_fredmd.md) and [volatility-analysis instructions](../../README_volatility_analysis.md). Existing results live in `output/results/` and `output/tables/`; new reporting artifacts live in `output/reports/`.

```text
# Evaluate saved forecast uncertainty, then refresh reports
python src/forecast_inference.py
python src/build_project_report.py
```

See [forecast uncertainty instructions](../../README_forecast_inference.md) for the three selected main comparisons, later-period robustness, inference assumptions and sensitivity settings. Report generation checks the inference input/output checksums and requires a fresh inference run if those files change.

## 9. Possible next research steps, not yet implemented

- Keep forecasting and contemporaneous explanation as separate research questions. Interpret the financial same-month results as an association, with the timing stated explicitly.
- A monthly log-volatility forecasting experiment would make the forecasting target transformation closer to the new primary specifications. The broad monthly FRED-MD forecasts currently use levels only.
- Predefine a small number of forecast combinations or factor-based macro models before evaluating new data. Do not select a specification simply because it happens to win among the inspected configurations.
- Before new confirmatory work, freeze model choices and the evaluation protocol. The selected forecast uncertainty analysis does not correct the earlier search across models, grids and periods.
- For genuine real-time forecasting, obtain historical vintages and publication dates. Conservative delays on a revised snapshot are an approximation, not a substitute for those data.
