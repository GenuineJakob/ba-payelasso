"""Run the current thesis scope with six readable Python scripts.

Install: python -m pip install tidyfinance==0.5.3 numpy pandas scipy scikit-learn matplotlib threadpoolctl
Run from Code: python current_scope/run.py
Quick check:   python current_scope/run.py --limit 2
Refresh data:  python current_scope/run.py --download
Reuse outputs: python current_scope/run.py --stage evaluate

Outputs go to output/current_scope; short runs use their own smoke directory.
The script folder contains code only. Cached inputs are the default. Fresh
downloads are separately stored and can differ from the revised data in the
thesis. The Paye-inspired target stays in levels. The two expanded pools differ
in both composition and size. Later evaluations and test families are retrospective.
"""
import sys
sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import pandas as pd
from importlib.metadata import PackageNotFoundError

import data
import evaluate
import forecast
from settings import PROJECT, RESAMPLES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=PROJECT, help="Code directory containing data/raw.")
    parser.add_argument("--output", type=Path, help="Separate output directory (default: Code/output/current_scope).")
    parser.add_argument("--stage", choices=["all", "data", "forecasts", "evaluate"], default="all")
    parser.add_argument("--experiment", choices=["all", "original", "paye", "expanded"], default="all")
    parser.add_argument("--limit", type=int, help="Positive number of origins per design for a separate smoke run.")
    parser.add_argument("--download", action="store_true", help="Use tidyfinance to refresh supported data in the output folder.")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    if args.download and args.stage not in {"all", "data"}:
        parser.error("--download requires --stage data or all")
    if args.stage == "evaluate" and (args.limit or args.experiment != "all"):
        parser.error("Inference requires full evaluations of all retained designs")
    output = args.output or args.project / "output" / "current_scope"
    if args.limit or args.experiment != "all":
        output = output / f"smoke_{args.experiment}_{args.limit or 'full'}"
    output.mkdir(parents=True, exist_ok=True)
    raw = args.project / "data" / "raw"
    inputs = [p for p in raw.iterdir() if p.name in {"sp500.csv", "fred_macro.csv", "goyal-welch-a.csv",
              "PayoutPaperTS_04_Mar_2021_monthly.csv", "fredmd_appendix0324.zip", "fredmd_snapshot_2026-09.csv"}]
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in inputs}
    if args.stage in {"all", "data"}:
        # Refreshes go to a separate folder so the supplied raw cache stays reproducible.
        refreshed = output / "raw" if args.download else None
        if refreshed:
            data.download(refreshed)
        base, expanded, pools, metadata = data.build(raw, refreshed)
        base.to_csv(output / "base.csv", index_label="date")
        expanded.to_csv(output / "expanded.csv", index_label="date")
        metadata.to_csv(output / "macro_dictionary.csv", index=False)
        (output / "pools.json").write_text(json.dumps(pools, indent=2) + "\n")
        provenance = dict(refreshed=bool(refreshed), source_sha256=hashes,
            artifacts={name: sha256((output/name).read_bytes()).hexdigest()
                       for name in ["base.csv", "expanded.csv", "pools.json", "macro_dictionary.csv"]},
            refreshed_source_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in refreshed.glob("*.csv")} if refreshed else {})
        (output / "data_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    else:
        provenance = json.loads((output / "data_manifest.json").read_text())
        for name, digest in provenance["artifacts"].items():
            if sha256((output/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Processed input changed: {name}; rerun --stage data.")
        base = pd.read_csv(output / "base.csv", index_col=0, parse_dates=True)
        expanded = pd.read_csv(output / "expanded.csv", index_col=0, parse_dates=True)
        pools = json.loads((output / "pools.json").read_text())
    if args.stage in {"all", "forecasts"}:
        predictions = forecast.run(base, expanded, pools, output, args.limit, args.experiment)
        saved = dict(complete=args.limit is None and args.experiment == "all",
                     data_sha256=sha256((output / "data_manifest.json").read_bytes()).hexdigest(),
                     predictions_sha256=sha256((output / "predictions.csv").read_bytes()).hexdigest())
        (output / "forecast_manifest.json").write_text(json.dumps(saved, indent=2) + "\n")
    if args.stage == "evaluate":
        predictions = pd.read_csv(output / "predictions.csv")
    # Infer from complete saved forecasts; short runs cannot support the thesis comparisons.
    if args.stage == "evaluate" or (args.stage == "all" and args.limit is None and args.experiment == "all"):
        saved = json.loads((output / "forecast_manifest.json").read_text())
        if not saved["complete"] or saved["data_sha256"] != sha256((output / "data_manifest.json").read_bytes()).hexdigest():
            raise ValueError("Inference requires complete forecasts from the current data build.")
        if saved["predictions_sha256"] != sha256((output / "predictions.csv").read_bytes()).hexdigest():
            raise ValueError("Saved forecasts changed; rerun --stage forecasts.")
        evaluate.run(predictions, base, output, args.resamples)
    if any(sha256(p.read_bytes()).hexdigest() != hashes[p.name] for p in inputs):
        raise RuntimeError("An original input changed during the run.")
    manifest = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                    settings=vars(args), source_sha256=hashes,
                    script_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                    software={name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl"]})
    try:
        manifest["software"]["tidyfinance"] = version("tidyfinance")
    except PackageNotFoundError:
        manifest["software"]["tidyfinance"] = "not installed; cached-data run"
    (output / f"run_{args.stage}.json").write_text(json.dumps(manifest, default=str, indent=2) + "\n")
    print(f"Saved outputs to {output}", flush=True)


if __name__ == "__main__":
    main()
