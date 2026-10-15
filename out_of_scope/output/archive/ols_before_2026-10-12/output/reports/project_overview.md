# Project overview: stock-market volatility, macroeconomics and Lasso

Report generated 2026-10-11T14:30:54+00:00. This overview describes the current code and saved experiments. The report builder reads existing outputs; it does not refit models or alter input data. The companion [results report](results_report.md) and [browsable page](results.html) contain the numerical comparisons.

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
| FRED-MD without implied volatility | 124 | Macro plus rates, exchange rates and stock indicators; excludes VIXCLSx |
| Full FRED-MD | 125 | Identical broad financial pool plus VIXCLSx |

FRED-MD transformations come from the pinned CSV's transformation row: level, first difference, second difference, log level, first log difference, second log difference, or change in fractional growth. They use only the current and earlier observations. Invalid logs and divisions become missing, with no future backfill or interpolation. [Transformation documentation](https://files.stlouisfed.org/files/htdocs/wp/2015/2015-012.pdf).

The complete variable names, descriptions, transformations, eligibility decisions and delays are available in the [monthly forecast feature dictionary](../../data/processed/fredmd_feature_metadata.csv), [same-month feature dictionary](../../data/processed/contemporaneous_feature_metadata.csv), and [quarterly feature dictionary](../../data/processed/quarterly_feature_metadata.csv).

For forecasting, broad FRED-MD macro indicators have a two-month assumed delay; rates, exchange rates, S&P prices and implied volatility use the origin month, while S&P dividend yield and P/E have a six-month delay. The same-month explanation removes these shifts and uses reference-month observations ex post. Original/Paye forecasts do not implement this explicit publication-delay schedule, so those saved outputs should be interpreted as revised-data comparisons rather than historical real-time forecasts.

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

The saved model comparisons describe realized sample performance. The current project does not report forecast-comparison significance tests, confidence intervals or causal identification. Many correlated predictors can make OLS rank deficient even when predictor count is below training sample size. In rank-deficient fits, forecasts can be computed but individual coefficients are not uniquely identified. Broad OLS is treated as a diagnostic comparison; increasing predictor count alone does not resolve this issue.

The fixed FRED-MD vintage and original cached macro data are revised data. Assumed reporting delays do not undo later revisions or reconstruct the information set at every historical date. The volatility source differs from Paye's excess-return construction; the stock index target is not implied volatility. A high contemporaneous VIX selection rate is consistent with a strong observed market relationship but is not evidence that VIX improves next-month forecasts.

The last full volatility-analysis validation records 45 passing tests and independent reconstruction of 13,758 new-analysis model predictions, including their training R-squared, error metrics and timing. The report additionally recalculates MSE, RMSE and AR-relative performance for all 238 canonical saved comparison rows. This report check is separate from refitting a model or testing statistical significance.

## 8. Files, entry points and how to reproduce

| Stage | Scripts under src/ | Purpose |
| --- | --- | --- |
| Raw input acquisition | download_data.py; download_fredmd.py | Existing price/macro CSVs; pinned FRED-MD CSV, appendix and checksum manifest |
| Dataset construction | build_dataset.py; build_fredmd_dataset.py; build_volatility_analysis_datasets.py | Original, delayed FRED-MD, same-month and quarterly datasets |
| Original models | train_models.py | Fixed and 360-month rolling benchmark/OLS/Lasso |
| Paye models | train_paye_monthly_individual.py; train_paye_monthly_kitchen_sink.py; train_paye_monthly_batch_experiments.py | Individual, joint and historical block/grid comparisons |
| Expanded models | train_fredmd.py | Four delayed monthly predictor pools |
| New questions | train_contemporaneous.py; train_quarterly.py | Same-month explanation and next-quarter forecasts in both scales |
| Shared estimation | partially_penalized_lasso.py; paye_monthly_utils.py; volatility_analysis_utils.py | Unpenalized controls, tuning, imputation, exports and metrics |
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

For definitions and detailed runner options, see [FRED-MD instructions](../../README_fredmd.md) and [volatility-analysis instructions](../../README_volatility_analysis.md). Existing results live in `output/results/` and `output/tables/`; new reporting artifacts live in `output/reports/`.

## 9. Possible next research steps, not yet implemented

- Keep forecasting and contemporaneous explanation as separate research questions. Interpret the financial same-month results as an association, with the timing stated explicitly.
- A monthly log-volatility forecasting experiment would make the forecasting target transformation closer to the new primary specifications. The broad monthly FRED-MD forecasts currently use levels only.
- Predefine a small number of forecast combinations or factor-based macro models before evaluating new data. Do not select a specification simply because it happens to win among the inspected configurations.
- If making inferential claims, add an appropriate time-series forecast-comparison design and uncertainty estimates; the current error tables alone do not establish significance.
- For genuine real-time forecasting, obtain historical vintages and publication dates. Conservative delays on a revised snapshot are an approximation, not a substitute for those data.
