"""Integrity, data reconstruction, metric reconciliation and rerun comparisons."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'current_scope'))
import data


def check_integrity():
    manifest = json.loads((ROOT / 'submission_manifest.json').read_text())
    for relative, expected in manifest['files_sha256'].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Submission file missing or changed: ' + relative)
    return len(manifest['files_sha256'])


def metric_check(folder):
    predictions = pd.read_csv(folder / 'predictions.csv')
    metrics = pd.read_csv(folder / 'metrics.csv')
    keys = ['experiment', 'pool', 'model']
    if predictions.duplicated(keys + ['date']).any() or metrics.duplicated(keys).any():
        raise ValueError('Duplicate forecast or metric rows in ' + str(folder))
    if not np.isfinite(predictions[['actual', 'prediction', 'benchmark']]).all().all():
        raise ValueError('Nonfinite forecasts in ' + str(folder))
    grouped = predictions.groupby(keys, sort=False)
    if len(grouped) != len(metrics):
        raise ValueError('Forecast groups do not match metric rows')
    for row in metrics.itertuples():
        group = grouped.get_group((row.experiment, row.pool, row.model))
        if len(group) != row.n:
            raise ValueError('Forecast count differs from metric table')
        mse = float(np.mean((group.actual - group.prediction)**2))
        ar_mse = float(np.mean((group.actual - group.benchmark)**2))
        np.testing.assert_allclose([mse, np.sqrt(mse), 1-mse/ar_mse],
                                   [row.MSE, row.RMSE, row.OOS_R2], rtol=1e-9, atol=2e-12)
        origins = pd.to_datetime(group.date)
        targets = pd.to_datetime(group.target_date)
        if not (targets == origins + pd.offsets.MonthEnd(1)).all():
            raise ValueError('Target dates must be one month after forecast origins')
    return len(predictions), len(metrics)


def check_reference():
    monthly = ROOT / 'results' / 'reference' / 'monthly'
    combined = ROOT / 'results' / 'reference' / 'combined'
    base, expanded, pools, metadata = data.build(ROOT / 'data' / 'raw')
    for name, rebuilt in [('base', base), ('expanded', expanded)]:
        saved = pd.read_csv(monthly / (name + '.csv'), index_col=0, parse_dates=True)
        if not rebuilt.index.equals(saved.index) or list(rebuilt.columns) != list(saved.columns):
            raise ValueError('Rebuilt data layout differs: ' + name)
        np.testing.assert_allclose(rebuilt.to_numpy(), saved.to_numpy(), rtol=0, atol=3e-12, equal_nan=True)
    if pools != json.loads((monthly / 'pools.json').read_text()):
        raise ValueError('Frozen predictor pools differ')
    monthly_n, monthly_metrics = metric_check(monthly)
    combined_n, combined_metrics = metric_check(combined)
    if monthly_n != 9213 or combined_n != 825:
        raise ValueError('The submission requires 9,213 monthly and 825 combined-pool forecast rows')
    software = json.loads((monthly / 'run_all.json').read_text())['software']
    from importlib.metadata import version
    for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'threadpoolctl']:
        if version(name) != software[name]:
            raise ValueError('Scientific package differs from thesis: ' + name)
    return dict(raw_data_reconstructed=True, baseline_predictors=len(pools['baseline']),
                macro_predictors=len(pools['macro']), combined_predictors=108,
                monthly_forecast_rows=monthly_n, combined_forecast_rows=combined_n,
                metric_rows_reconciled=monthly_metrics + combined_metrics,
                scientific_package_versions_match=True)


def compare_predictions(generated, reference, smoke=False):
    actual = pd.read_csv(generated / 'predictions.csv')
    expected = pd.read_csv(reference / 'predictions.csv')
    keys = ['experiment', 'pool', 'model', 'date']
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError('Duplicate forecast comparison keys')
    actual = actual.set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()
    if smoke:
        if not actual.index.isin(expected.index).all():
            raise ValueError('Smoke forecast keys absent from reference')
        expected = expected.loc[actual.index]
    elif not actual.index.equals(expected.index):
        raise ValueError('Full rerun has different forecast dates/models from the reference')
    for column in ['target_date', 'pool_size']:
        if not (actual[column] == expected[column]).all():
            raise ValueError('Forecast metadata differs: ' + column)
    for column in ['actual', 'benchmark', 'prediction']:
        np.testing.assert_allclose(actual[column], expected[column], rtol=0, atol=2e-9)
    # Each artifact is recomputed by the corresponding runner; the reference
    # files remain untouched. A platform can introduce tiny BLAS differences.
    return dict(rows_compared=len(actual),
                maximum_prediction_difference=float(np.max(np.abs(actual.prediction - expected.prediction))),
                absolute_prediction_tolerance=2e-9,
                reconciled_metric_rows=metric_check(generated)[1])
