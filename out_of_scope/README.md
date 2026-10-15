# Material outside the thesis scope

Moved here on 15 October 2026. Nothing was deleted. Paths below mirror their previous paths relative to `Code/`; see `move_manifest.json` for the complete inventory and checksums.

Included: contemporaneous explanation (levels/logs), quarterly forecasts, options-only VIX robustness and its raw inputs, Paye batch block/penalty-grid experiments, smoke outputs, superseded snapshots, development caches, and comprehensive project reports/refresh tools spanning those experiments.

The retained thesis still uses the original monthly models, Paye individual/joint exercise, macro-only expansion, combined-pool extension and forecast inference. Shared raw sources, monthly datasets and mixed FRED-MD output files remain in the active project unchanged.

`tests/test_monthly_model_integration.py` is the preserved original mixed test file; the active copy excludes its batch-specific import and test. Archived files otherwise retain their bytes. Archived scripts and reports are historical artifacts: their embedded root assumptions, manifests and relative links refer to the former layout, so this archive is not a standalone runnable project. Use the manifest to restore a workflow together with its dependencies before running it. Shared dependencies remain in the active `Code/src/` and `Code/data/` folders.

To undo a move, restore its `destination` to its `source` from the manifest, checking for conflicts first. Restore the original mixed test file if restoring batch tests. Do not overwrite newer active files.
