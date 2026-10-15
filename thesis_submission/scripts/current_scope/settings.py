"""The thesis's retained monthly designs, collected in one place."""
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
# The six free controls are current volatility and its five preceding months.
AR = ["volatility"] + [f"vol_lag_{i}" for i in range(1, 6)]
ORIGINAL = ["unemployment", "fed_funds", "default_spread", "term_spread",
            "commercial_paper_spread", "inflation", "ip_growth", "gdp_growth",
            "ppi_volatility", "ip_volatility"]
PAYE = {"cp": "commercial_paper_spread", "dfr": "dfr", "dfy": "default_spread",
        "ip": "ip_growth", "ipvol": "ip_volatility", "npy": "npy",
        "ppivol": "ppi_volatility", "tms": "term_spread"}
BASELINE = [x for x in ORIGINAL if x not in {"gdp_growth", "ppi_volatility", "ip_volatility"}] + ["ip_volatility", "ppi_volatility"]
DELAYED = {"unemployment", "inflation", "ip_growth", "ppi_volatility", "ip_volatility"}
FRED = ["UNRATE", "FEDFUNDS", "CPIAUCSL", "INDPRO", "AAA", "BAA", "GS10",
        "TB3MS", "CP3M", "CPN3M", "GDPC1", "PPIACO"]
VINTAGE = "2026-09"
# Retain activity, credit and price groups; exclude rate/exchange-rate and equity groups.
GROUPS = {1, 2, 3, 4, 5, 7}
ALPHAS = np.logspace(-5, 1, 200)[::-1]
PERIODS = {"main": ("2001-01-31", "2020-12-31"),
           "later": ("2021-01-31", "2023-11-30")}
SEED = 20261013
RESAMPLES = 20000

