"""Training-window standardized least squares with an explicit SVD cutoff.

Rank is numerical rank at the recorded cutoff, not a statement about exact
mathematical dependence. Returned coefficients use the original feature units.
"""
from importlib.metadata import version
import platform
import sys

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


OLS_CONFIGURATION = {
    "solver": "numpy.linalg.lstsq", "center_predictors": True,
    "scale_predictors": "training population standard deviation (ddof=0)",
    "rcond_policy": "eps(float64) * max(n_training_rows, n_fitted_predictors)",
    "intercept": "unpenalized, recovered after centering",
    "coefficient_units": "original predictors",
    "rank_interpretation": "numerical rank of centered, standardized predictors at recorded rcond",
}


def software_versions():
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            **{package: version(package) for package in
               ["numpy", "scipy", "pandas", "scikit-learn", "threadpoolctl"]}}


class TrainingScaledOLS(RegressorMixin, BaseEstimator):
    """Fit unpenalized scalar-response OLS without a library-default tolerance."""

    def fit(self, X, y):
        values = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if values.ndim != 2 or not values.shape[1] or target.shape != (len(values),) or not len(values):
            raise ValueError("OLS needs a nonempty 2D design and one response per row.")
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            raise ValueError("OLS inputs must be finite; fit predictor imputation on training rows first.")
        self.n_features_in_ = values.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.mean_ = values.mean(axis=0)
        centered = values - self.mean_
        constant = np.ptp(values, axis=0) == 0
        centered[:, constant] = 0
        self.scale_ = np.sqrt(np.mean(centered**2, axis=0))
        self.scale_[constant] = 1
        standardized = centered / self.scale_
        self.rcond_ = float(np.finfo(np.float64).eps * max(standardized.shape))
        target_mean = target.mean()
        beta, _, self.rank_, self.singular_ = np.linalg.lstsq(
            standardized, target - target_mean, rcond=self.rcond_
        )
        beta[constant] = 0
        self.coef_ = beta / self.scale_
        self.intercept_ = float(target_mean - self.mean_ @ self.coef_)
        # Keep raw-design diagnostics to expose scale-sensitive truncation.
        raw_singular = np.linalg.svd(centered, compute_uv=False)
        self.raw_singular_ = raw_singular
        self.raw_rank_conventional_ = int(np.sum(raw_singular > self.rcond_ * raw_singular[0]))
        self.raw_rank_1e6_ = int(np.sum(raw_singular > 1e-6 * raw_singular[0]))
        return self

    def predict(self, X):
        if hasattr(self, "feature_names_in_") and hasattr(X, "columns"):
            X = X.loc[:, self.feature_names_in_]
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_ or not np.isfinite(values).all():
            raise ValueError("Prediction inputs must be finite and match fitted columns.")
        return values @ self.coef_ + self.intercept_

    def diagnostics(self):
        def condition(s):
            return float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
        return {
            "ols_rank": int(self.rank_), "ols_fitted_columns": self.n_features_in_,
            "ols_rank_deficient": bool(self.rank_ < self.n_features_in_),
            "ols_solver": OLS_CONFIGURATION["solver"], "ols_standardized": True,
            "ols_rcond": self.rcond_, "ols_singular_cutoff": float(self.rcond_ * self.singular_[0]),
            "ols_scaled_smin": float(self.singular_[-1]), "ols_scaled_smax": float(self.singular_[0]),
            "ols_scaled_condition_number": condition(self.singular_),
            "ols_raw_condition_number": condition(self.raw_singular_),
            "ols_raw_rank_conventional": self.raw_rank_conventional_,
            "ols_raw_rank_at_1e6": self.raw_rank_1e6_,
        }
