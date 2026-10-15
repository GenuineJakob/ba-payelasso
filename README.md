# Thesis code and data

The active project supports the monthly level-volatility thesis: Paye-inspired individual and joint models, original fixed/rolling OLS and Lasso, and the nine-variable baseline versus 99-series macro-only pool. The supplemental 108-predictor combined run remains in the writing workspace's `combined_pool_run/`.

## Active files

- `current_scope/`: consolidated thesis runner; `python current_scope/run.py --help` lists stages and output options. Its default output is `output/current_scope/`.
- `src/`: original monthly data builders, downloaders, estimators, forecasting runners and forecast inference. `conceptual_ar6_lasso_example.py` explains the same AR/Lasso design.
- `data/raw/` and `data/processed/`: frozen inputs and monthly datasets required by those workflows.
- `output/results/`, `output/tables/`, and `output/reports/forecast_inference.md`: saved monthly evidence and numerical provenance.
- `tests/` and `notebooks/01_eda.ipynb`: retained estimator/monthly checks and monthly volatility exploration.

Shared FRED-MD data, metadata and output CSVs include additional financial-pool fields/rows. They remain intact because the thesis consumes the same files and verifies their hashes. Keeping those containers does not add those other pools to thesis scope. The original OLS-correction manifests also remain as numerical provenance; historical archive paths now resolve through `out_of_scope/`.

## Archived material

`out_of_scope/` holds separate same-month explanation, quarterly forecasts, options-only VIX robustness, Paye block/grid experiments, development smoke runs, superseded snapshots and comprehensive reports covering those branches. Its folder structure mirrors the original project. `out_of_scope/move_manifest.json` records each original and destination path and SHA-256 checksum.

The former comprehensive `src/build_project_report.py` and cross-experiment `src/rerun_ols.py` are archived with those workflows. The thesis manuscript's original references to them document the earlier project layout. Monthly model production, `src/forecast_inference.py`, and all frozen thesis results remain available at their original paths. No models or saved forecasts were changed by the reorganization.

Existing Git staging was left untouched. The mixed monthly integration test keeps its original-model, individual/joint and conceptual-example checks; its original full version is preserved in the archive, including the batch experiment test.
