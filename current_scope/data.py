"""Build only the target, original/Paye predictors and two expanded pools.

Cached inputs reproduce the thesis. Optional refreshes use tidyfinance 0.5.3
for Yahoo, FRED and the pinned FRED-MD vintage and go to a separate directory.
The author-hosted annual CSV is retained: tidyfinance's Goyal-Welch endpoint
does not return the corporate bond-return field needed by this thesis.
"""
from io import BytesIO
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from settings import AR, BASELINE, DELAYED, FRED, GROUPS, ORIGINAL, VINTAGE


def monthly(frame):
    expected = pd.date_range(frame.index[0], frame.index[-1], freq="ME")
    if not frame.index.is_unique or not frame.index.equals(expected):
        raise ValueError("Data must have unique, consecutive month ends.")


def download(destination):
    """Refresh supported sources without overwriting the thesis's raw cache."""
    import tidyfinance as tf
    tf.set_backend("pandas")
    destination.mkdir(parents=True, exist_ok=True)
    print("Downloading Yahoo prices, FRED series and pinned FRED-MD...", flush=True)
    prices = tf.download_data("Stock Prices", symbols=["^GSPC"],
                              start_date="1970-01-01", end_date="2023-12-31")
    fred = tf.download_data("FRED", series=FRED,
                            start_date="1970-01-01", end_date="2023-12-31")
    macro = tf.download_data("FRED", "FRED-MD", vintage=VINTAGE, transform=True)
    if prices.empty or fred.empty or macro.empty:
        raise ValueError("A requested download returned no observations.")
    missing = set(FRED) - set(fred.series)
    if missing:
        raise ValueError(f"FRED download omitted series: {sorted(missing)}")
    prices[["date", "adjusted_close"]].rename(columns={"adjusted_close": "Close"}).to_csv(
        destination / "sp500.csv", index=False)
    fred.pivot(index="date", columns="series", values="value").to_csv(destination / "fred_macro.csv")
    macro.to_csv(destination / "fredmd_transformed.csv", index=False)


def transform(x, code):
    """Seven official FRED-MD transformations; never use future observations."""
    x = x.where(np.isfinite(x))
    if code == 1:
        result = x
    elif code in (2, 3):
        result = x.diff() if code == 2 else x.diff().diff()
    elif code in (4, 5, 6):
        result = np.log(x.where(x > 0))
        for _ in range(code - 4):
            result = result.diff()
    elif code == 7:
        result = (x / x.shift().where(x.shift() != 0) - 1).diff()
    else:
        raise ValueError(f"Unknown FRED-MD transformation: {code}")
    return result.where(np.isfinite(result))


def build(raw, refreshed=None):
    """Return base data, expanded data, frozen pools and their descriptions."""
    raw = Path(raw)
    source = Path(refreshed) if refreshed else raw
    manifest = json.loads((raw / f"fredmd_sources_{VINTAGE}.json").read_text())
    for name in [f"fredmd_snapshot_{VINTAGE}.csv", "fredmd_appendix0324.zip"]:
        if sha256((raw / name).read_bytes()).hexdigest() != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"Pinned source checksum differs: {name}")
    prices = pd.read_csv(source / "sp500.csv")
    dates = pd.to_datetime(prices.iloc[:, 0], errors="coerce")
    price = pd.Series(pd.to_numeric(prices.Close, errors="coerce").to_numpy(), index=dates)
    price = price.loc[price.index.notna()].dropna().sort_index()
    if price.index.has_duplicates or (price <= 0).any():
        raise ValueError("Prices must be positive with unique dates.")
    vol = np.sqrt(12 * np.log(price).diff().pow(2).resample("ME").sum()).rename("volatility")
    m = pd.read_csv(source / "fred_macro.csv", index_col=0, parse_dates=True).resample("ME").last().ffill()
    x = pd.DataFrame(index=m.index)
    x["unemployment"], x["fed_funds"] = m.UNRATE, m.FEDFUNDS
    x["default_spread"], x["term_spread"] = m.BAA - m.AAA, m.GS10 - m.TB3MS
    x["commercial_paper_spread"] = m.CPN3M.combine_first(m.CP3M) - m.TB3MS
    for name, code in [("inflation", "CPIAUCSL"), ("ip_growth", "INDPRO"), ("gdp_growth", "GDPC1")]:
        x[name] = np.log(m[code]).diff(12)
    x["ip_volatility"] = x.ip_growth.rolling(12).std()
    x["ppi_volatility"] = np.log(m.PPIACO).diff(12).rolling(12).std()
    annual = pd.read_csv(raw / "goyal-welch-a.csv")
    annual.index = pd.to_datetime(annual.yyyy.astype(str) + "-12-31")
    x["dfr"] = (annual.corprate - annual.ltrate).resample("ME").last().ffill().reindex(x.index)
    payout = pd.read_csv(raw / "PayoutPaperTS_04_Mar_2021_monthly.csv", sep=";", decimal=",")
    payout.index = pd.to_datetime(dict(year=payout.year, month=payout.month, day=1)) + pd.offsets.MonthEnd(0)
    x["npy"] = payout.net_pay_yld.resample("ME").last().reindex(x.index)
    base = pd.concat([vol, x], axis=1)
    base["target"] = base.volatility.shift(-1)
    base = base.dropna(subset=["volatility", "target", "unemployment", "fed_funds",
                               "default_spread", "term_spread", "inflation", "ip_growth"])
    monthly(base)
    for lag in range(1, 6):
        base[f"vol_lag_{lag}"] = base.volatility.shift(lag)
    expanded = base[AR + ["target"]].copy()
    for name in BASELINE:
        expanded[f"core__{name}"] = base[name].shift(2 if name in DELAYED else 0)
    with ZipFile(raw / "fredmd_appendix0324.zip") as archive:
        dictionary = pd.read_csv(BytesIO(archive.read("FRED-MD Appendix/FRED-MD_updated_appendix.csv")), encoding="cp1252")
    dictionary.index = dictionary.fred.str.casefold()
    if not dictionary.index.is_unique:
        raise ValueError("The official dictionary contains duplicate series codes.")
    if refreshed:
        fredmd = pd.read_csv(source / "fredmd_transformed.csv")
        fredmd.index = pd.to_datetime(fredmd.pop("date")) + pd.offsets.MonthEnd(0)
        fredmd = fredmd.drop(columns="vintage")
        codes = {}
    else:
        snapshot = pd.read_csv(raw / f"fredmd_snapshot_{VINTAGE}.csv")
        codes = snapshot.iloc[0, 1:].astype(int).to_dict()
        fredmd = snapshot.iloc[1:].set_index("sasdate").apply(pd.to_numeric)
        fredmd.index = pd.to_datetime(fredmd.index) + pd.offsets.MonthEnd(0)
        fredmd = pd.DataFrame({name: transform(fredmd[name], codes[name]) for name in fredmd})
    monthly(fredmd)
    rows, features = [], {}
    initial = expanded.loc[:"2000-12-31"].tail(240).index
    if len(initial) != 240 or initial[0] != pd.Timestamp("1981-01-31"):
        raise ValueError("Initial eligibility window must be January 1981--December 2000.")
    for name in fredmd:
        info = dictionary.loc[name.casefold()]
        if int(info.group) not in GROUPS:
            continue
        feature = f"fredmd__{name}"
        values = fredmd[name].shift(2).reindex(base.index)
        coverage = np.isfinite(values.reindex(initial)).mean()
        eligible = coverage >= .95
        if eligible:
            features[feature] = values
        rows.append(dict(feature=feature, source_code=name, description=info.description,
                         group=int(info.group), tcode=codes.get(name), delay=2,
                         initial_coverage=coverage, eligible=eligible))
    metadata = pd.DataFrame(rows)
    expanded = pd.concat([expanded, pd.DataFrame(features, index=base.index)], axis=1)
    pools = {"baseline": [f"core__{name}" for name in BASELINE],
             "macro": metadata.loc[metadata.eligible, "feature"].tolist()}
    if len(pools["macro"]) != 99 or len(pools["baseline"]) != 9:
        raise ValueError("The retained scope requires nine baseline and 99 macro features.")
    return base, expanded, pools, metadata

