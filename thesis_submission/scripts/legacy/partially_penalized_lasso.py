"""Lasso with explicitly unpenalized controls and chronological tuning.

Only the additional predictors enter the L1 penalty. Each validation fold
estimates its own projections and residual scaling from its training rows.
Returned coefficients and predictions use the original feature units.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, lars_path, lasso_path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def training_medians(X: pd.DataFrame) -> pd.Series:
    """Fit medians on training rows, keeping entirely missing features at zero.

    All-missing features become constant zero in training and therefore have
    zero slope. Keeping their columns preserves coefficient/export alignment.
    """
    return pd.Series(
        {name: X[name].median() if X[name].notna().any() else 0.0 for name in X},
        dtype=float,
    )


@dataclass
class _ControlPath:
    coefficients: np.ndarray
    intercepts: np.ndarray
    target_on_controls: LinearRegression
    predictors_on_controls: LinearRegression
    scaler: StandardScaler
    imputation_values: pd.Series | None = None
    relative_dual_gaps: np.ndarray | None = None
    polished_alphas_n: int = 0


class PartiallyPenalizedLassoCV(RegressorMixin, BaseEstimator):
    """Select alpha using complete model predictions, keeping controls free.

    ``X`` is a DataFrame containing both controls and additional predictors.
    For each alpha, residualized, standardized additional predictors are
    penalized; the intercept and control coefficients are recovered by OLS.
    Controls stay in the equation even if all additional coefficients are zero.
    """

    def __init__(
        self,
        controls: list[str],
        alphas: np.ndarray,
        n_splits: int = 5,
        max_iter: int = 50000,
        tol: float = 1e-6,
        impute_missing: bool = False,
        solver: str = "sklearn",
    ):
        self.controls = controls
        self.alphas = alphas
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.tol = tol
        self.impute_missing = impute_missing
        self.solver = solver

    def _fit_path(
        self, X: pd.DataFrame, y: np.ndarray, alphas: np.ndarray
    ) -> _ControlPath:
        imputation_values = None
        if self.impute_missing:
            imputation_values = training_medians(X[self.penalized_features_])
            X = X.fillna(imputation_values)
        controls = X[self.controls]
        additional = X[self.penalized_features_]
        target_on_controls = LinearRegression().fit(controls, y)
        target_residuals = y - target_on_controls.predict(controls)
        predictors_on_controls = LinearRegression().fit(controls, additional)
        predictor_residuals = (
            additional.to_numpy() - predictors_on_controls.predict(controls)
        )

        # A predictor in the span of the controls has no additional signal.
        # Do not magnify projection roundoff by standardizing it to unit size.
        roundoff = (
            np.finfo(float).eps
            * max(X.shape)
            * np.maximum(1.0, np.linalg.norm(additional.to_numpy(), axis=0))
        )
        redundant = np.linalg.norm(predictor_residuals, axis=0) <= roundoff
        predictor_residuals[:, redundant] = 0.0
        scaler = StandardScaler().fit(predictor_residuals)
        standardized = scaler.transform(predictor_residuals)
        gap_scale = max(float(np.mean(target_residuals ** 2)), np.finfo(float).eps)
        polished_alphas_n = 0
        if self.solver == "lars":
            # LARS' knot calculations are more reliable with unit-size target
            # residuals. beta=s*b and alpha'=alpha/s preserve the objective;
            # all exported coefficients are converted back immediately.
            target_scale = np.sqrt(gap_scale)
            normalized_target = target_residuals / target_scale
            normalized_alphas = alphas / target_scale
            # The Lasso path is piecewise linear. LARS finds its knots without
            # thousands of coordinate sweeps through correlated predictors.
            # Interpolating those knots evaluates the SAME requested alpha grid.
            path_alphas, _, path_coefficients = lars_path(
                standardized, normalized_target, method="lasso",
                alpha_min=float(normalized_alphas.min()), max_iter=self.max_iter,
            )
            standardized_coefficients = target_scale * np.array([
                np.interp(normalized_alphas, path_alphas[::-1], values[::-1])
                for values in path_coefficients
            ])
            errors = target_residuals[:, None] - standardized @ standardized_coefficients
            correlations = np.max(np.abs(standardized.T @ errors), axis=0) / len(X)
            dual = errors * np.minimum(1.0, alphas / np.maximum(correlations, np.finfo(float).tiny))
            primal = np.mean(errors ** 2, axis=0) / 2 + alphas * np.abs(standardized_coefficients).sum(axis=0)
            gaps = (
                primal - np.mean(target_residuals[:, None] * dual, axis=0)
                + np.mean(dual ** 2, axis=0) / 2
            )
            # Nearly dependent columns can make a LARS knot inaccurate. Check
            # the objective, not only the solver's stop flag, and polish just
            # those requested alphas from their existing near-optimal solutions.
            inaccurate = gaps > self.tol * gap_scale
            if inaccurate.any():
                polished_alphas_n = int(inaccurate.sum())
                initial = standardized_coefficients[:, np.flatnonzero(inaccurate)[0]]
                _, polished, polished_gaps = lasso_path(
                    standardized, normalized_target, alphas=normalized_alphas[inaccurate],
                    coef_init=initial / target_scale, tol=self.tol, max_iter=self.max_iter,
                )
                standardized_coefficients[:, inaccurate] = polished * target_scale
                gaps[inaccurate] = polished_gaps * target_scale ** 2
            if (gaps > 1.1 * self.tol * gap_scale).any():
                worst = int(np.argmax(gaps))
                raise RuntimeError(
                    f"Lasso path failed the dual-gap check: n={len(X)}, p={standardized.shape[1]}, "
                    f"alpha={alphas[worst]:.8g}, relative_gap={gaps[worst] / gap_scale:.8g}. "
                    "No valid forecast can be exported."
                )
        else:
            _, standardized_coefficients, gaps = lasso_path(
                standardized,
                target_residuals,
                alphas=alphas,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        additional_coefficients = standardized_coefficients / scaler.scale_[:, None]

        # With beta fixed, OLS gives the unpenalized intercept and AR block.
        adjusted_targets = y[:, None] - additional.to_numpy() @ additional_coefficients
        control_model = LinearRegression().fit(controls, adjusted_targets)
        coefficients = np.zeros((X.shape[1], len(alphas)))
        coefficients[X.columns.get_indexer(self.controls)] = control_model.coef_.T
        coefficients[X.columns.get_indexer(self.penalized_features_)] = additional_coefficients
        return _ControlPath(
            coefficients=coefficients,
            intercepts=np.asarray(control_model.intercept_),
            target_on_controls=target_on_controls,
            predictors_on_controls=predictors_on_controls,
            scaler=scaler,
            imputation_values=imputation_values,
            relative_dual_gaps=np.maximum(gaps, 0) / gap_scale,
            polished_alphas_n=polished_alphas_n,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        if self.solver not in {"sklearn", "lars"}:
            raise ValueError("solver must be 'sklearn' or 'lars'.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame with named controls and predictors.")
        if not X.columns.is_unique:
            raise ValueError("Predictor columns must be unique.")
        if not self.controls or len(set(self.controls)) != len(self.controls):
            raise ValueError("Provide a nonempty list of unique unpenalized controls.")
        missing = set(self.controls).difference(X.columns)
        if missing:
            raise ValueError(f"Missing unpenalized controls: {sorted(missing)}")
        self.penalized_features_ = [name for name in X.columns if name not in self.controls]
        if not self.penalized_features_:
            raise ValueError("Lasso requires at least one additional predictor.")
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or len(target) != len(X):
            raise ValueError("y must be one-dimensional with one value per row of X.")
        values = X.to_numpy(dtype=float)
        if not np.isfinite(X[self.controls].to_numpy(dtype=float)).all() or not np.isfinite(target).all():
            raise ValueError("Unpenalized controls and target must be finite.")
        if np.isinf(values).any() or (not self.impute_missing and np.isnan(values).any()):
            raise ValueError("Training predictors must be finite unless missing-value imputation is enabled.")
        alphas = np.asarray(self.alphas, dtype=float)
        if alphas.ndim != 1 or not len(alphas) or not np.isfinite(alphas).all() or (alphas <= 0).any():
            raise ValueError("alphas must be a nonempty sequence of finite positive values.")
        self.alphas_ = np.unique(alphas)[::-1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.mse_path_ = np.empty((len(self.alphas_), self.n_splits))
        self.relative_dual_gap_path_ = np.empty_like(self.mse_path_)
        self.polished_cv_alphas_n_ = 0

        for fold, (train_rows, validation_rows) in enumerate(
            TimeSeriesSplit(n_splits=self.n_splits).split(X)
        ):
            path = self._fit_path(X.iloc[train_rows], target[train_rows], self.alphas_)
            self.relative_dual_gap_path_[:, fold] = path.relative_dual_gaps
            self.polished_cv_alphas_n_ += path.polished_alphas_n
            validation = X.iloc[validation_rows]
            if path.imputation_values is not None:
                validation = validation.fillna(path.imputation_values)
            predictions = (
                validation.to_numpy() @ path.coefficients + path.intercepts
            )
            self.mse_path_[:, fold] = np.mean(
                (target[validation_rows, None] - predictions) ** 2, axis=0
            )

        # Descending alphas make ties prefer the stronger penalty.
        best = int(np.argmin(self.mse_path_.mean(axis=1)))
        self.alpha_ = float(self.alphas_[best])
        fitted = self._fit_path(X, target, np.array([self.alpha_]))
        self.coef_ = fitted.coefficients[:, 0]
        self.intercept_ = float(fitted.intercepts[0])
        self.target_on_controls_ = fitted.target_on_controls
        self.predictors_on_controls_ = fitted.predictors_on_controls
        self.scaler_ = fitted.scaler
        self.imputation_values_ = fitted.imputation_values
        self.relative_dual_gap_ = float(fitted.relative_dual_gaps[0])
        self.selected_cv_relative_dual_gap_ = float(self.relative_dual_gap_path_[best].max())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["coef_", "intercept_", "alpha_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a DataFrame with named features.")
        features = X.loc[:, self.feature_names_in_]
        if self.imputation_values_ is not None:
            features = features.fillna(self.imputation_values_)
        values = features.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Prediction features must be finite.")
        return values @ self.coef_ + self.intercept_
