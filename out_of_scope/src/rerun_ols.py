"""Refresh saved FRED-MD/new-analysis OLS outputs without refitting AR/Lasso.

Archives prior files before updating predictions, coefficients, diagnostics,
metrics and run metadata. Raw/processed data and other model predictions stay
unchanged. Usage: python src/rerun_ols.py
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fredmd_config import PROJECT_ROOT, RESULTS_DIR, TABLES_DIR
from stable_ols import OLS_CONFIGURATION, software_versions
from train_fredmd import load_experiment, fit_ols_window, evaluate_forecasts
from volatility_analysis_utils import load_analysis, scale_volatility, fit_ols, evaluate_predictions


def read_csv(path, dated=True):
    return pd.read_csv(path, float_precision="round_trip", parse_dates=["date"] if dated else False)


def set_fields(frame, mask, updates):
    if int(mask.sum()) != 1:
        raise ValueError("Each OLS fit must have exactly one saved diagnostic/coefficient row.")
    for name, value in updates.items():
        if name not in frame:
            frame[name] = pd.Series(index=frame.index, dtype="object" if isinstance(value, (str, bool)) else float)
        frame.loc[mask, name] = value


def archive_file(path, archive, hashes):
    relative = path.relative_to(PROJECT_ROOT)
    destination = archive / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError("Archive contains a different run; choose a fresh --archive-dir.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    hashes[str(relative)] = digest


def refresh_case(kind, label, data, metadata, build, archive, archived_hashes):
    paths = {
        "predictions": RESULTS_DIR / f"{kind}_predictions_{label}.csv",
        "coefficients": RESULTS_DIR / f"{kind}_coefficients_{label}.csv",
        "diagnostics": TABLES_DIR / f"{kind}_window_diagnostics_{label}.csv",
        "metrics": TABLES_DIR / f"{kind}_model_comparison_{label}.csv",
        "manifest": RESULTS_DIR / f"{kind}_run_manifest_{label}.json",
    }
    for path in paths.values():
        archive_file(path, archive, archived_hashes)
    predictions = read_csv(paths["predictions"])
    before = predictions.copy(deep=True)
    coefficients = read_csv(paths["coefficients"])
    diagnostics = read_csv(paths["diagnostics"])
    if kind == "fredmd":
        frame, window = data, build["window_size"]
        ols_mask = predictions.model == "ols"
    else:
        scale = label.split("_", 1)[0]
        frame = scale_volatility(data, build["controls"], scale)
        window = build["window_size"]
        ols_mask = (predictions.model == "joint_ols") | predictions.model.str.startswith("individual__")
    coefficient_groups = coefficients.groupby(["date", "pool", "model"]).indices
    ranks, conventional_raw_ranks, raw_legacy_ranks, conditions = [], [], [], []
    for index, row in predictions.loc[ols_mask].iterrows():
        position = frame.index.get_loc(row.date)
        train, test = frame.iloc[position-window:position], frame.iloc[[position]]
        features = build["pools"][row.pool]
        if row.model.startswith("individual__"):
            features = [row.model.removeprefix("individual__")]
        dmask = (diagnostics.date == row.date) & (diagnostics.pool == row.pool)
        if kind == "fredmd":
            prediction, beta, numerical = fit_ols_window(train, test, features)
            c_mask = (coefficients.date == row.date) & (coefficients.pool == row.pool) & (coefficients.model == row.model)
            set_fields(coefficients, c_mask, beta.to_dict())
        else:
            fit = fit_ols(train, test, build["controls"], features)
            prediction = fit["prediction"]
            beta = pd.concat([pd.Series({"intercept": fit["intercept"]}), fit["coefficients"]])
            positions = coefficient_groups[(row.date, row.pool, row.model)]
            names = coefficients.loc[positions, "feature"]
            if set(names) != set(beta.index):
                raise ValueError("Exported coefficient names differ from the OLS design.")
            coefficients.loc[positions, "coefficient"] = names.map(beta).to_numpy()
            numerical = {k: v for k, v in fit.items() if k not in ["prediction", "coefficients", "intercept", "alpha"]}
            dmask &= diagnostics.model == row.model
            predictions.loc[index, "training_R2"] = fit["training_R2"]
            predictions.loc[index, "n_selected"] = fit["n_selected"]
            predictions.loc[index, "diagnostic_only"] = bool(fit["ols_rank_deficient"])
            numerical["diagnostic_only"] = bool(fit["ols_rank_deficient"])
        set_fields(diagnostics, dmask, numerical)
        predictions.loc[index, "prediction"] = prediction
        ranks.append(numerical["ols_rank"])
        conventional_raw_ranks.append(numerical["ols_raw_rank_conventional"])
        raw_legacy_ranks.append(numerical["ols_raw_rank_at_1e6"])
        conditions.append(numerical["ols_scaled_condition_number"])
    # These stored models are deliberately not reestimated in this correction.
    pd.testing.assert_frame_equal(predictions.loc[~ols_mask], before.loc[~ols_mask])
    pd.testing.assert_series_equal(predictions.actual, before.actual)
    pd.testing.assert_series_equal(predictions.benchmark, before.benchmark)
    metrics = evaluate_forecasts(predictions) if kind == "fredmd" else evaluate_predictions(predictions)
    predictions.to_csv(paths["predictions"], index=False)
    coefficients.to_csv(paths["coefficients"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    manifest = json.loads(paths["manifest"].read_text())
    correction = {
        "refreshed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software_versions": software_versions(), "ols_configuration": OLS_CONFIGURATION,
        "ols_fits_reestimated": len(ranks), "AR_and_Lasso": "saved predictions reused unchanged",
        "original_run_software_versions": manifest.get("software_versions", "not recorded"),
        "archive": str(archive.relative_to(PROJECT_ROOT)),
        "prior_files_sha256": {k: archived_hashes[str(p.relative_to(PROJECT_ROOT))] for k, p in paths.items()},
    }
    manifest["ols_refresh"] = correction
    manifest["ols_configuration"] = OLS_CONFIGURATION
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
    # Check the physical CSV round trip, not only the in-memory unchanged rows.
    saved = read_csv(paths["predictions"])
    pd.testing.assert_frame_equal(saved.loc[~ols_mask], before.loc[~ols_mask])
    print(f"Updated {kind} {label}: {len(ranks)} OLS fits; rank range {min(ranks)}–{max(ranks)}", flush=True)
    return {"kind": kind, "label": label, "ols_fits": len(ranks),
            "rank_min": min(ranks), "rank_max": max(ranks),
            "raw_conventional_rank_min": min(conventional_raw_ranks),
            "raw_1e6_rank_min": min(raw_legacy_ranks), "scaled_condition_max": max(conditions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    archive = args.archive_dir or PROJECT_ROOT / "output" / "archive" / f"ols_before_{stamp}"
    archive = archive.resolve()
    if not archive.is_relative_to(PROJECT_ROOT):
        raise ValueError("Archive directory must be within the project.")
    reports, archived_hashes = [], {}
    with threadpool_limits(limits=1):
        for kind in ["fredmd", "contemporaneous", "quarterly"]:
            data, metadata, build = load_experiment() if kind == "fredmd" else load_analysis(kind)
            for path in sorted(RESULTS_DIR.glob(f"{kind}_predictions_*.csv")):
                label = path.stem.removeprefix(f"{kind}_predictions_")
                reports.append(refresh_case(kind, label, data, metadata, build, archive, archived_hashes))
    summary = {"software_versions": software_versions(), "ols_configuration": OLS_CONFIGURATION,
               "cases": reports, "archive": str(archive.relative_to(PROJECT_ROOT)),
               "archived_sha256": archived_hashes}
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "ols_refresh_archive_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (TABLES_DIR / "ols_correction_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
