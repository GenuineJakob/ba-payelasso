"""Cache Cboe's options-based VIX history; never refresh it implicitly."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import urlopen

from download_fredmd import sha256_file
from fredmd_config import RAW_DIR

URL = "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
FILENAME = "cboe_vix_history.csv"
MANIFEST = "cboe_vix_sources.json"


def cache_history(raw_dir=RAW_DIR, source_file=None):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path, provenance = raw_dir / FILENAME, raw_dir / MANIFEST
    if path.exists() or provenance.exists():
        if not (path.exists() and provenance.exists()):
            raise ValueError("Incomplete VIX cache: both CSV and source manifest are required.")
        info = json.loads(provenance.read_text())
        if info["url"] != URL or info["sha256"] != sha256_file(path):
            raise ValueError("VIX cache checksum or source mismatch.")
        return path
    if source_file is None:
        with urlopen(URL, timeout=30) as response:
            content = response.read()
    else:
        content = Path(source_file).read_bytes()
    # Validate schema and observations before creating a cache.
    from io import BytesIO
    import pandas as pd
    from train_vix_robustness import monthly_vix
    daily = pd.read_csv(BytesIO(content))
    monthly_vix(daily)
    path.write_bytes(content)
    provenance.write_text(json.dumps({
        "url": URL, "file": FILENAME, "sha256": sha256_file(path),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_date": str(daily.DATE.iloc[0]), "last_date": str(daily.DATE.iloc[-1]),
        "interpretation": "Options-based VIX; early history back-calculated under modern methodology, not a historical real-time vintage. No realized-volatility extension.",
    }, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, help="Import a CSV downloaded from the stated Cboe URL.")
    args = parser.parse_args()
    print(cache_history(source_file=args.source_file))
