"""Download/cache the pinned FRED-MD snapshot and its official appendix.

Usage: python src/download_fredmd.py [--refresh]
No API key or additional data-download package is required.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

from fredmd_config import (
    APPENDIX_NAME, APPENDIX_URL, RAW_DIR, SNAPSHOT_NAME, SNAPSHOT_URL,
    SOURCE_MANIFEST_NAME, VINTAGE,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_source(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response:
                return response.read()
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            print(f"Retrying the same pinned URL (attempt {attempt + 2}/3)", flush=True)
    raise RuntimeError("Download did not return content.")


def validate_download(name: str, contents: bytes) -> None:
    if name == SNAPSHOT_NAME:
        frame = pd.read_csv(StringIO(contents.decode("utf-8-sig")))
        if frame.shape[1] != 127 or frame.columns[0] != "sasdate":
            raise ValueError("Pinned FRED-MD snapshot must contain sasdate and 126 predictors.")
        if str(frame.iloc[0, 0]).strip() != "Transform:":
            raise ValueError("Missing FRED-MD transformation row.")
    else:
        with ZipFile(BytesIO(contents)) as archive:
            if "FRED-MD Appendix/FRED-MD_updated_appendix.csv" not in archive.namelist():
                raise ValueError("Official appendix archive lacks the variable dictionary.")
            if archive.testzip() is not None:
                raise ValueError("Corrupt FRED-MD appendix archive.")


def download_fredmd(raw_dir: Path = RAW_DIR, refresh: bool = False) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / SOURCE_MANIFEST_NAME
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    artifacts = {}
    for name, url in [(SNAPSHOT_NAME, SNAPSHOT_URL), (APPENDIX_NAME, APPENDIX_URL)]:
        path = raw_dir / name
        if path.exists() and not refresh:
            contents = path.read_bytes()
            old = previous.get("artifacts", {}).get(name)
            if old and (old["url"] != url or old["sha256"] != hashlib.sha256(contents).hexdigest()):
                raise ValueError(f"Cached source failed integrity check: {path}. Use --refresh to redownload.")
            print(f"Using cached {name}", flush=True)
        else:
            print(f"Downloading {url}", flush=True)
            contents = fetch_source(url)
        validate_download(name, contents)
        if not path.exists() or refresh:
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(contents)
            temporary.replace(path)
        artifacts[name] = {"url": url, "sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "vintage": VINTAGE,
        "retrieved_at_utc": previous.get("retrieved_at_utc") if previous and not refresh else datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Source manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Redownload the same pinned vintage.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    download_fredmd(args.raw_dir, args.refresh)


if __name__ == "__main__":
    main()
