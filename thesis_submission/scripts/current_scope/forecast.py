"""One forecasting loop for the retained fixed and rolling experiments."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data import monthly
from models import lasso, ols
from settings import AR, ORIGINAL, PAYE, PERIODS


def designs(base, expanded, pools):
    original = base.dropna(subset=AR + ORIGINAL + ["target"])
    paye = base.dropna(subset=AR + list(PAYE.values()) + ["target"])
    yield "original_fixed", original, {"original": ORIGINAL}, None, PERIODS["main"], False
    yield "original_rolling", original, {"original": ORIGINAL}, 360, PERIODS["main"], False
    yield "paye", paye, {**{name: [feature] for name, feature in PAYE.items()},
                         "joint": list(PAYE.values())}, 240, PERIODS["main"], False
    for period, dates in PERIODS.items():
        yield f"expanded_{period}", expanded, pools, 240, dates, True


def run(base, expanded, pools, output, limit=None, selected="all"):
    """Write forecasts, coefficients, windows, metrics and selection frequencies."""
    rows, coefficient_rows, window_rows = [], [], []
    with threadpool_limits(limits=1):
        for name, data, candidates, window, dates, broad in designs(base, expanded, pools):
            if selected != "all" and not name.startswith(selected):
                continue
            if broad:
                monthly(data)
            positions = [i for i, date in enumerate(data.index)
                         if pd.Timestamp(dates[0]) <= date <= pd.Timestamp(dates[1])
                         and (window is None or i >= window)]
            if broad and len(positions) != (240 if name.endswith("main") else 35):
                raise ValueError("Expanded evaluations must retain every declared origin.")
            if limit:
                positions = positions[:limit]
            print(f"{name}: {len(positions)} origins; pools { {k: len(v) for k, v in candidates.items()} }", flush=True)
            fixed = {}
            for number, i in enumerate(positions, 1):
                test = data.iloc[[i]]
                # Exclude the forecast origin; rolling windows contain only earlier training rows.
                train = data.loc[:"2000-12-31"] if window is None else data.iloc[i-window:i]
                date = test.index[0]
                if train.empty or train.index[-1] >= date:
                    raise ValueError("Training origins must strictly precede the forecast origin.")
                if not np.isfinite(train[AR + ["target"]]).all().all():
                    raise ValueError("Do not impute targets or volatility controls.")
                # Cache the fixed fit once; rolling designs estimate new coefficients at each origin.
                ar_fit = fixed.get("ar") or ols(train[AR], train.target)
                if window is None:
                    fixed["ar"] = ar_fit
                benchmark = float(ar_fit.predict(test[AR])[0])
                for pool, extra in candidates.items():
                    columns = AR + extra
                    if window is None and pool in fixed:
                        fits = fixed[pool]
                    else:
                        fits = {"ar": ar_fit, "ols": ols(train[columns], train.target, scaled=broad),
                                "lasso": lasso(train[columns], train.target, solver="lars" if broad else "cd")}
                        if window is None:
                            fixed[pool] = fits
                    common = dict(experiment=name, pool=pool, date=date,
                                  target_date=date + pd.offsets.MonthEnd(1),
                                  actual=float(test.target.iloc[0]), benchmark=benchmark)
                    for model, fit in fits.items():
                        used = AR if model == "ar" else columns
                        prediction = float(fit.predict(test[used])[0])
                        # Count only penalised extras, excluding the six free volatility controls.
                        selected_count = int((np.abs(fit.coefficients[6:]) > 1e-12).sum()) if model == "lasso" else 0
                        rows.append({**common, "model": model, "prediction": prediction,
                                     "alpha": fit.alpha, "n_selected": selected_count,
                                     "pool_size": len(extra)})
                        coefficient_rows.append({**common, "model": model, "intercept": fit.intercept,
                                                 "alpha": fit.alpha, **dict(zip(used, fit.coefficients))})
                    window_rows.append({**common, "training_start": train.index[0],
                        "training_end": train.index[-1], "training_n": len(train),
                        "training_missing": int(train[extra].isna().sum().sum()),
                        "forecast_missing": int(test[extra].isna().sum().sum()),
                        "lasso_relative_gap": fits["lasso"].gap})
                if number == 1 or number % 24 == 0 or number == len(positions):
                    print(f"  {number}/{len(positions)} through {date.date()}", flush=True)
    predictions = pd.DataFrame(rows)
    if predictions.empty or not np.isfinite(predictions[["actual", "prediction", "benchmark"]]).all().all():
        raise ValueError("No valid forecasts were produced.")
    coefficients = pd.DataFrame(coefficient_rows)
    predictions.to_csv(output / "predictions.csv", index=False)
    coefficients.to_csv(output / "coefficients.csv", index=False)
    pd.DataFrame(window_rows).to_csv(output / "windows.csv", index=False)
    metrics = []
    for (name, pool, model), g in predictions.groupby(["experiment", "pool", "model"], sort=False):
        mse = float(np.mean((g.actual - g.prediction)**2))
        reference = float(np.mean((g.actual - g.benchmark)**2))
        metrics.append(dict(experiment=name, pool=pool, model=model, n=len(g),
                            first_origin=g.date.min(), last_origin=g.date.max(), MSE=mse,
                            RMSE=np.sqrt(mse), OOS_R2=1-mse/reference))
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    frequencies = []
    for (name, pool), g in coefficients.loc[coefficients.model == "lasso"].groupby(["experiment", "pool"], sort=False):
        extra = pools[pool] if name.startswith("expanded") else (ORIGINAL if name.startswith("original")
                else (list(PAYE.values()) if pool == "joint" else [PAYE[pool]]))
        for feature in extra:
            frequencies.append(dict(experiment=name, pool=pool, feature=feature,
                selected_n=int((g[feature].abs() > 1e-12).sum()), selected_share=float((g[feature].abs() > 1e-12).mean())))
    pd.DataFrame(frequencies).to_csv(output / "selection.csv", index=False)
    return predictions
