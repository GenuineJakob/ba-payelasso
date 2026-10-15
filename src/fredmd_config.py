"""Frozen data and experiment settings for the FRED-MD extension."""

from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "output" / "results"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
VINTAGE = "2026-09"
SNAPSHOT_NAME = f"fredmd_snapshot_{VINTAGE}.csv"
APPENDIX_NAME = "fredmd_appendix0324.zip"
SOURCE_MANIFEST_NAME = f"fredmd_sources_{VINTAGE}.json"
SNAPSHOT_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    f"fred-md/monthly/{VINTAGE}-md.csv"
)
APPENDIX_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-md/fred-md_appendix0324.zip"
)
DATASET_NAME = "fredmd_dataset.csv"
METADATA_NAME = "fredmd_feature_metadata.csv"
BUILD_MANIFEST_NAME = "fredmd_build_manifest.json"
BASE_DATA_PATH = PROCESSED_DIR / "dataset.csv"
AR_CONTROLS = ["vol"] + [f"vol_lag_{lag}" for lag in range(1, 6)]
TARGET = "target"
WINDOW_SIZE = 240
COVERAGE_THRESHOLD = 0.95
ALPHAS = np.logspace(-5, 1, 200)
PERIODS = {
    "main": ("2001-01-31", "2020-12-31"),
    "holdout": ("2021-01-31", "2023-11-30"),
}
MACRO_GROUPS = {1, 2, 3, 4, 5, 7}
BASELINE_FEATURES = {
    "unemployment": ("Unemployment rate", 2, 2, "existing level"),
    "fed_funds": ("Federal funds rate", 6, 0, "existing level"),
    "default_spread": ("BAA minus AAA yield", 6, 0, "existing yield spread"),
    "term_spread": ("10-year minus 3-month Treasury yield", 6, 0, "existing yield spread"),
    "commercial_paper_spread": ("3-month commercial paper minus Treasury yield", 6, 0, "existing yield spread"),
    "inflation": ("Consumer-price inflation", 7, 2, "existing 12-month log change"),
    "ip_growth": ("Industrial-production growth", 1, 2, "existing 12-month log change"),
    "ip_volatility": ("Industrial-production growth volatility", 1, 2, "existing 12-month rolling standard deviation"),
    "ppi_volatility": ("Producer-price inflation volatility", 7, 2, "existing 12-month rolling standard deviation"),
}
POOL_NAMES = ["current_monthly", "fredmd_macro", "fredmd_no_iv", "fredmd_full"]
DATA_CAVEAT = (
    "Fixed September 2026 revised-data snapshot with assumed reporting delays; "
    "not a reconstruction of historical real-time information. "
    "Target is annualized realized volatility in levels, not Paye's log volatility."
)
