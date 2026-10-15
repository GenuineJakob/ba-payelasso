# Combined baseline + macro forecasting run

The combined pool contains all nine monthly baseline columns and all 99 eligible FRED-MD macro columns: 108 additional candidate predictors plus six unpenalized volatility controls and an intercept. All additional predictors are penalized equally in Lasso.

This is a nested predictor expansion of the existing nine-variable baseline. Related economic series and alternative transformations are intentionally retained; columns are not removed because they are correlated.

The frozen September 2026 inputs, predictor-specific delays, 240-month rolling windows, five chronological validation folds and 200-value penalty grid are unchanged. Medians, projections and scaling are fitted only on each training fold. The main sample has 240 origins (January 2001–December 2020); the later sample has 35 origins (January 2021–November 2023). The target is next-month annualized realized volatility in levels.

## Forecast accuracy

Positive MSE reduction means improvement over the matching AR forecast. MSE is in squared level-volatility units.

| Period | Pool | Model | MSE | RMSE | MSE reduction vs AR |
| --- | --- | --- | ---: | ---: | ---: |
| Main | baseline | OLS | 0.008811203 | 0.093868006 | -9.404% |
| Main | baseline | LASSO | 0.008048904 | 0.089715686 | +0.061% |
| Main | macro | OLS | 0.022539062 | 0.150130150 | -179.856% |
| Main | macro | LASSO | 0.007959520 | 0.089216144 | +1.171% |
| Main | combined | OLS | 0.035516726 | 0.188458818 | -340.992% |
| Main | combined | LASSO | 0.007945975 | 0.089140196 | +1.339% |
| Later | baseline | OLS | 0.002424150 | 0.049235658 | -45.213% |
| Later | baseline | LASSO | 0.001834619 | 0.042832452 | -9.899% |
| Later | macro | OLS | 0.012803728 | 0.113153559 | -666.978% |
| Later | macro | LASSO | 0.001669104 | 0.040854673 | +0.016% |
| Later | combined | OLS | 0.015889400 | 0.126053163 | -851.818% |
| Later | combined | LASSO | 0.001669104 | 0.040854673 | +0.016% |

## Combined Lasso: matched comparisons

| Period | Reference | MSE reduction | 95% paired block bootstrap interval | HAC p | Holm-adjusted p |
| --- | --- | ---: | --- | ---: | ---: |
| Main | AR | +1.339% | [-0.137%, +3.079%] | 0.276 | 0.827 |
| Main | baseline LASSO | +1.279% | [-0.232%, +3.052%] | 0.283 | 0.827 |
| Main | macro LASSO | +0.170% | [-0.321%, +0.544%] | 0.493 | 0.827 |
| Main | combined OLS | +77.628% | [+40.289%, +91.308%] | 0.139 | 0.555 |
| Later | AR | +0.016% | [-0.000%, +0.062%] | 0.277 | 0.554 |
| Later | baseline LASSO | +9.022% | [-2.574%, +22.418%] | 0.160 | 0.479 |
| Later | macro LASSO | +0.000% | [-0.000%, +0.000%] | — | — |
| Later | combined OLS | +89.495% | [+68.111%, +94.669%] | 0.039 | 0.154 |

Inference is exploratory and retrospective. Intervals use 20,000 paired circular-block resamples with six-month blocks, with three- and twelve-month sensitivity results saved separately. HAC tests are two-sided tests of mean loss differences. Holm adjustment is applied separately to the four new comparisons in each period; it does not revise the thesis's existing test families. The later sample is not an untouched confirmation sample. These intervals resample saved forecast losses and do not refit or retune models. The later combined-versus-macro forecasts are numerically identical, so a loss-difference test is not reported for that comparison.

## Predictor selection

Main: zero extra predictors in 202/240 windows; mean selected extras 0.263; maximum 10.

- fredmd__RPI: 37/240 windows (15.42%).
- core__inflation: 3/240 windows (1.25%).
- fredmd__HWIURATIO: 3/240 windows (1.25%).
- fredmd__USCONS: 3/240 windows (1.25%).
- fredmd__USWTRADE: 3/240 windows (1.25%).
- fredmd__M1SL: 3/240 windows (1.25%).
- fredmd__UEMPMEAN: 2/240 windows (0.83%).
- fredmd__OILPRICEx: 2/240 windows (0.83%).
- fredmd__DSERRG3M086SBEA: 2/240 windows (0.83%).
- fredmd__IPNCONGD: 1/240 windows (0.42%).

Later: zero extra predictors in 33/35 windows; mean selected extras 0.057; maximum 1.

- fredmd__HWI: 2/35 windows (5.71%).

## Validation and reproducibility

- Frozen inputs and original method files match the recorded thesis hashes; scientific package versions match the existing run.
- Independent original-estimator checks agree for both OLS and Lasso at the first main and first later origins (absolute prediction tolerance 2e-9); selected Lasso penalties agree.
- All 275 origins are retained; targets and comparison dates match the existing baseline and macro runs. Recomputed AR benchmarks agree within 2e-12; machine-precision CSV round-trip differences are aligned only in comparison metadata, without changing forecast predictions. All predictions are finite and coefficient exports reconstruct the saved forecasts.
- Every original input and method file is checked again after the run to confirm unchanged contents.

Run `run_combined.py` with the original project's scientific Python environment. `run_manifest.json` records source hashes, software, method settings and artifact hashes. Forecasts, coefficients, window diagnostics, selections, accuracy comparisons and loss inference are saved alongside this report.
