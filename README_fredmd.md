# FRED-MD volatility experiments

This extension compares an unpenalized AR(6) benchmark with AR(6)+OLS and
AR(6)+Lasso using a frozen September 2026 FRED-MD snapshot. Only additional
predictors are penalized. The target remains next-month annualized realized
volatility in levels.

From the `Code` directory, create a persistent Python environment if needed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-fredmd.txt
```

Download and prepare the separate experiment dataset:

```bash
python src/download_fredmd.py
python src/build_fredmd_dataset.py
```

The downloader uses the same pinned vintage on every run, requires no API key,
and verifies cached checksums. `--refresh` redownloads that same vintage.
The existing `data/processed/dataset.csv` is the source for volatility and
baseline predictors and is never overwritten.

The extension uses scikit-learn's LARS Lasso path on the large, correlated
predictor pools. Its piecewise-linear path is evaluated at the unchanged
200-value alpha grid, with dual gaps checked against a relative tolerance of
1e-6. Previous experiments retain ordinary coordinate descent. No additional
solver package is needed.
Target residuals are normalized internally for numerical conditioning, with
the alpha grid rescaled equivalently and coefficients restored to original
units. The volatility target and the penalized objective remain the same.

Run a short check, then the full experiments:

```bash
python src/train_fredmd.py --period all --max-forecasts 3
python src/train_fredmd.py --period main
python src/train_fredmd.py --period holdout
```

`--period all` runs both complete periods. The default is `main`.
Short checks write separate `_smoke` files. Previous Paye and original-model
outputs are preserved. Rerunning an extension experiment replaces its own
corresponding `fredmd_` outputs.

## Fixed predictor pools

- `current_monthly`: nine existing genuinely monthly predictors. Excludes
  repeated annual/quarterly observations and predictors that stop in 2019.
- `fredmd_macro`: official groups 1–5 and 7.
- `fredmd_no_iv`: all eligible FRED-MD series except `VIXCLSx`.
- `fredmd_full`: the same pool plus `VIXCLSx`.

The snapshot has 126 candidate series. Eligibility requires at least 95%
finite observations in the January 1981–December 2000 training window,
after transformations and reporting delays. This selects 99 macroeconomic
and 125 total FRED-MD predictors in the pinned data; `ACOGNO` is excluded
because its history begins in 1992. The financial pools differ only by
the extended `VIXCLSx` volatility proxy. Eligibility is never recalculated using holdout coverage.

Official transformation codes come from the CSV's first row. Macroeconomic
indicators have an assumed two-month delay. Interest rates, exchange rates,
S&P 500 prices, and the extended volatility proxy have no additional delay. Dividend
yield and price/earnings ratio have a six-month delay. Metadata records every
transformation, group, delay, coverage decision, and source checksum.

## Evaluation and missing values

The main period uses forecast origins January 2001–December 2020. The later
holdout uses January 2021–November 2023. A date labels the **forecast origin**;
`target_date` is the following month. Specifications are fixed before the
holdout, but 240-month rolling estimation and five-fold chronological alpha
tuning continue using observations available at each origin.

Each training window fits its own predictor medians. Within Lasso tuning,
medians, AR projections, and scaling are fitted separately in each training
fold. Calendar months are never deleted because of predictor gaps. Entirely
missing or constant predictors in a fold have zero coefficients. AR controls
and targets must remain finite and are never imputed.

This is a **revised-data, fixed-snapshot experiment with assumed reporting
delays**, not a reconstruction of historical real-time information. The
historical definition of the spliced `VIXCLSx` series matters: its pre-1986
extension uses normalized realized daily-return volatility from Bloom, not
options-implied observations. Later history contains options-based VXO/VIX;
FRED-MD records the replacement in its December 2021 vintage without specifying
every observation-level splice boundary. The cached series begins in July 1962.
Do not interpret the entire history as observed implied volatility. More
predictors do not guarantee better forecasts.

The options-only robustness runner addresses **same-month explanation**,
not a reestimation of these next-month forecasts. It uses Cboe's VIX from 1990
and matching AR(6) benchmarks. See [the analysis instructions](README_volatility_analysis.md)
and [the detailed results](output/reports/results_report.md).

## Outputs and checks

`output/results/fredmd_*` contains predictions, original-unit coefficients,
imputation values for windows with gaps, and the complete run specification.
Coefficient columns belonging to other pools are intentionally blank.

`output/tables/fredmd_*` contains MSE, RMSE, out-of-sample R-squared,
selection frequencies, imputation rates, and window diagnostics including OLS
rank and Lasso convergence warnings. OLS centers and standardizes every fitted
predictor using only its training window, then calls `numpy.linalg.lstsq` with
the explicit relative cutoff `eps(float64) * max(n_train, n_predictors)`.
Coefficients and intercepts are restored to the original feature units.
Rank is numerical rank at that recorded cutoff, not proof of mathematical
dependence. Diagnostics also report condition numbers, singular-value cutoffs,
and raw-design ranks under conventional and `1e-6` cutoffs. The earlier low
broad-pool ranks arose from scale-sensitive truncation; the corrected joint
fits have full numerical rank, although their forecast performance remains poor.
Run manifests record Python/package versions and the OLS solver specification.
An OLS count in `n_selected` is the predictor pool size, not variable selection.
Window diagnostics also record dual gaps for the fitted Lasso, its selected
cross-validation alpha, and the complete cross-validation path.

To recalculate only saved OLS fits and retain the existing AR/Lasso predictions:

```bash
python src/rerun_ols.py
python src/build_project_report.py
```

The OLS refresh archives prior outputs under `output/archive/` before updating
them and records its own package versions. It does not assign those versions
to inherited AR/Lasso runs whose original versions were not recorded. The
2026-10-12 correction is archived in `output/archive/ols_before_2026-10-12/`.

```bash
python -m pytest tests -q -p no:cacheprovider
```

Sources: [official FRED-MD downloads](https://www.stlouisfed.org/research/economists/mccracken/fred-databases)
and [McCracken and Ng, transformation documentation](https://files.stlouisfed.org/files/htdocs/wp/2015/2015-012.pdf).
Solver: [scikit-learn LARS Lasso path](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lars_path.html).


## Selected forecast uncertainty

For the three selected thesis comparisons, run `python src/forecast_inference.py`
then `python src/build_project_report.py`. This reads saved forecasts without
refitting, checks pairing and source-table MSEs, and exports unconditional
Giacomini–White/HAC mean-loss tests, Holm adjustments, paired circular-block
percentile intervals and dependence-setting sensitivities. The later sample
is separate retrospective robustness; the fixed model is a degenerate diagnostic.
See [README_forecast_inference.md](README_forecast_inference.md) and
[the standalone report](output/reports/forecast_inference.md).
