# Thesis submission: scripts, raw data and results

This folder reproduces the thesis's monthly S&P 500 volatility analysis from the included raw data. It includes the original fixed and rolling OLS/Lasso models, the Paye-inspired individual/joint exercise, the nine-variable baseline, the 99-series macro-only pool, and the additional 108-predictor combined pool. Quarterly, contemporaneous and options-only VIX experiments are outside this submission.

## Start on Windows

1. Extract the whole folder from the ZIP to a writable location. Do not run it from inside the ZIP.
2. Have **64-bit Python 3.12 or newer** installed. The command `py -3 --version` or `python3 --version` shows the installed version. The recorded scientific packages require a recent Python 3.
3. Double-click **START_WINDOWS.bat**.

The launcher creates a private `.venv` inside this folder, installs the pinned packages in `requirements.txt`, checks the supplied files, and starts the full analysis. No administrator rights, manual package installation, account, API key or data download is needed. Internet access is needed to install packages the first time; the analysis itself uses only the included raw data. Keep the console open until completion. The full run fits thousands of rolling models and can take a while; progress and logs show which stage is running. The console stays open at the end.

Alternatively, open a terminal in this folder and run:

```text
py -3 install_and_run.py
```

If the Python command on the computer is `python3` or `python`, substitute it for `py -3`. The same Python launcher also works on macOS and Linux with `python3 install_and_run.py`.

## Optional quick checks

```text
py -3 install_and_run.py --mode verify
py -3 install_and_run.py --mode smoke
```

`verify` checks every submitted file, rebuilds the monthly datasets from raw inputs, checks the frozen predictor pools and reconciles all 51 saved metric rows against 10,038 saved forecasts. It does not refit models. `smoke` fits the first two origins of every retained design, including the combined pool, and compares those predictions with the saved results. Running the launcher without options always performs the **full** refit and inference.

## Folder contents

| Path | Contents |
| --- | --- |
| `START_WINDOWS.bat` | Windows entry point; finds Python and starts installation/analysis |
| `install_and_run.py` | Creates/checks the private environment and installs dependencies |
| `requirements.txt` | Pinned scientific and supporting Python dependencies |
| `run_thesis.py` | Coordinates integrity checks, monthly fitting, combined fitting and result comparisons |
| `scripts/current_scope/` | Six thesis modules with brief explanatory comments; calculations unchanged |
| `scripts/legacy/` | Original OLS and partially penalized Lasso implementations used for independent combined-pool checks |
| `scripts/run_combined.py` | Combined-pool runner with package-relative input/output paths |
| `scripts/verify_submission.py` | Raw-data reconstruction, integrity and numerical comparison checks |
| `data/raw/` | All eight retained raw/source-provenance files, including the payout workbook and modeled CSV |
| `results/reference/monthly/` | Submitted monthly forecasts, coefficients, metrics, processed datasets, inference, figures and original provenance |
| `results/reference/combined/` | Submitted combined-pool forecasts, coefficients, metrics, comparisons, inference and report |
| `submission_manifest.json` | Checksums covering the submitted code, data, instructions and reference results |
| `VALIDATION.md` | Checks completed before packaging, including the full reproduction |

The program creates new results in `results/reproduced/<run time>/monthly/` and `results/reproduced/<run time>/combined/`. Smoke results go to `results/smoke/`. Each run writes console logs and `validation.json` to `logs/<run time>_<mode>/`. It never replaces `results/reference/`.

For the monthly results, start with `metrics.csv` and `inference_summary.csv`; `predictions.csv` and `coefficients.csv` contain the underlying forecasts and fits. For the combined extension, start with `report.md`, `model_comparison.csv` and `inference_summary.csv`. CSV reduction fields are fractions unless the column explicitly says percent; multiply fractions by 100 for percentages.

## Reproducibility details

The frozen September 2026 FRED-MD snapshot, appendix/dictionary, original price/macro sources, annual return data and payout data are supplied. New data are not downloaded. The monthly runner reconstructs its inputs from `data/raw/`; the combined run consumes those newly reconstructed monthly inputs. The methods preserve 240-month expanded/Paye windows, the 360-month original rolling design, five chronological tuning folds, the original penalty grid and 20,000-resample loss inference. Later-period comparisons remain retrospective.

The six core modules include brief explanatory comments; their executable code and the two independent estimators are unchanged. The combined runner adds package-relative paths, explicit input/output arguments and brief explanatory comments; its estimators, settings, inference and validation are retained. The installation/orchestration/verification files are submission wrappers. The included `run_all.json` preserves the original method hashes under `original_script_sha256` and identifies the supplied commented modules under `script_sha256`. It and the combined `run_manifest.json` retain the original result provenance: historical absolute paths in these JSON records are descriptive only and are never used to find files on the examiner's computer.

The run checks that scientific package versions match those recorded in the thesis. It compares every newly produced forecast against the reference, using an absolute tolerance of `2e-9` to allow small platform-dependent numerical differences. Checksums validate file integrity; they do not establish historical real-time data availability or statistical significance. Full results are only marked successful after all fitting, inference and validation stages finish.

If setup fails, check the displayed error, Internet access and the Python version. Use a full 64-bit CPython installation with `venv` and `pip` support. This folder must be writable. An old, incompatible `.venv` can be removed and recreated by starting the launcher again; do not remove the supplied data or reference results.
