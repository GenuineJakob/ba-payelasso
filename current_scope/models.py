"""OLS and partially penalized Lasso; AR coefficients and intercept stay free."""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, lars_path, lasso_path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from settings import ALPHAS


@dataclass
class Fit:
    intercept: float
    coefficients: np.ndarray
    medians: np.ndarray
    alpha: float = np.nan
    gap: float = np.nan

    def predict(self, x):
        values = np.asarray(x, dtype=float)
        values = np.where(np.isnan(values), self.medians, values)
        if not np.isfinite(values).all():
            raise ValueError("Prediction inputs must be finite after training-only imputation.")
        return self.intercept + values @ self.coefficients


def impute(x):
    """Only extra predictors can be filled; never impute AR controls or targets."""
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x[:, :6]).all() or np.isinf(x).any():
        raise ValueError("Six AR controls must be finite; extra predictors cannot be infinite.")
    medians = np.zeros(x.shape[1])
    for j in range(6, x.shape[1]):
        finite = x[:, j][np.isfinite(x[:, j])]
        medians[j] = np.median(finite) if len(finite) else 0
    return np.where(np.isnan(x), medians, x), medians


def ols(x, y, scaled=False):
    x, medians = impute(x)
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Targets must be finite.")
    if not scaled:
        model = LinearRegression().fit(x, y)
        return Fit(float(model.intercept_), model.coef_, medians)
    mean = x.mean(axis=0)
    centered = x - mean
    constant = np.ptp(x, axis=0) == 0
    centered[:, constant] = 0
    scale = np.sqrt(np.mean(centered**2, axis=0))
    scale[constant] = 1
    z = centered / scale
    beta = np.linalg.lstsq(z, y - y.mean(), rcond=np.finfo(float).eps * max(z.shape))[0]
    beta[constant] = 0
    beta /= scale
    return Fit(float(y.mean() - mean @ beta), beta, medians)


def penalty_path(x, y, alphas, solver):
    """Fit one training fold, including projection, scaling and imputation."""
    x, medians = impute(x)
    controls, extra = x[:, :6], x[:, 6:]
    y = np.asarray(y, dtype=float)
    target_model = LinearRegression().fit(controls, y)
    residual_y = y - target_model.predict(controls)
    extra_model = LinearRegression().fit(controls, extra)
    residual_x = extra - extra_model.predict(controls)
    roundoff = np.finfo(float).eps * max(x.shape) * np.maximum(1, np.linalg.norm(extra, axis=0))
    residual_x[:, np.linalg.norm(residual_x, axis=0) <= roundoff] = 0
    scaler = StandardScaler().fit(residual_x)
    z = scaler.transform(residual_x)
    gap_scale = max(float(np.mean(residual_y**2)), np.finfo(float).eps)
    if solver == "lars":
        target_scale = np.sqrt(gap_scale)
        knots, _, path = lars_path(z, residual_y / target_scale, method="lasso",
                                   alpha_min=float(alphas.min() / target_scale), max_iter=50000)
        beta = target_scale * np.array([np.interp(alphas / target_scale, knots[::-1], row[::-1]) for row in path])
        error = residual_y[:, None] - z @ beta
        correlation = np.max(np.abs(z.T @ error), axis=0) / len(x)
        dual = error * np.minimum(1, alphas / np.maximum(correlation, np.finfo(float).tiny))
        gap = (np.mean(error**2, axis=0) / 2 + alphas * np.abs(beta).sum(axis=0)
               - np.mean(residual_y[:, None] * dual, axis=0) + np.mean(dual**2, axis=0) / 2)
        repair = gap > 1e-6 * gap_scale
        if repair.any():
            initial = beta[:, np.flatnonzero(repair)[0]] / target_scale
            _, polished, polished_gap = lasso_path(z, residual_y / target_scale,
                alphas=alphas[repair] / target_scale, coef_init=initial, tol=1e-6, max_iter=50000)
            beta[:, repair], gap[repair] = polished * target_scale, polished_gap * target_scale**2
        if (gap > 1.1e-6 * gap_scale).any():
            raise RuntimeError("Lasso path did not converge to the declared objective tolerance.")
    else:
        _, beta, gap = lasso_path(z, residual_y, alphas=alphas, tol=1e-6, max_iter=50000)
    beta = beta / scaler.scale_[:, None]
    free = LinearRegression().fit(controls, y[:, None] - extra @ beta)
    coefficients = np.vstack([free.coef_.T, beta])
    return coefficients, np.asarray(free.intercept_), medians, np.maximum(gap, 0) / gap_scale


def lasso(x, y, solver="cd"):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape[1] <= 6 or not np.isfinite(y).all():
        raise ValueError("Lasso needs six complete controls, extra predictors and a finite target.")
    losses = []
    for train, validation in TimeSeriesSplit(n_splits=5).split(x):
        beta, intercept, medians, _ = penalty_path(x[train], y[train], ALPHAS, solver)
        values = np.where(np.isnan(x[validation]), medians, x[validation])
        predictions = values @ beta + intercept
        if not np.isfinite(predictions).all():
            raise ValueError("Validation predictions must be finite.")
        losses.append(np.mean((y[validation, None] - predictions)**2, axis=0))
    # Descending penalties make ties prefer the strongest penalty.
    alpha = float(ALPHAS[np.argmin(np.mean(losses, axis=0))])
    beta, intercept, medians, gap = penalty_path(x, y, np.array([alpha]), solver)
    return Fit(float(intercept[0]), beta[:, 0], medians, alpha, float(gap[0]))

