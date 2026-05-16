"""Bronze Layer — Raw Ingestion.

Reads the supplied flat files (CSV) **as-is** with no transformations beyond
canonical column renaming (which is purely cosmetic and reversible), then
persists them as parquet inside ``data/bronze/`` for fast downstream access.

A manifest (``data/bronze/_manifest.json``) records row counts, column lists,
and SHA-256 hashes of the source files for full auditability.

Run as a script::

    python -m src.ingestion.bronze_ingestion
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config
from src.utils.io import (
    file_sha256,
    normalize_columns,
    read_csv_resilient,
    setup_logging,
    write_manifest,
    write_parquet,
)

logger = logging.getLogger(__name__)


def _ingest_one(name: str, source_csv: Path, out_dir: Path) -> dict:
    """Ingest a single CSV into Bronze as parquet. Returns a manifest entry."""
    if not source_csv.exists():
        logger.warning("source file missing for '%s': %s", name, source_csv)
        return {
            "dataset": name,
            "source_path": str(source_csv),
            "status": "MISSING",
        }

    raw_df = read_csv_resilient(source_csv)
    original_cols = list(raw_df.columns)

    # Only normalize column names. Values are untouched — Bronze is raw.
    df = normalize_columns(raw_df)

    out_path = out_dir / f"{name}.parquet"
    write_parquet(df, out_path)

    return {
        "dataset": name,
        "source_path": str(source_csv),
        "source_sha256": file_sha256(source_csv),
        "rows": int(len(df)),
        "columns_original": original_cols,
        "columns_canonical": list(df.columns),
        "bronze_path": str(out_path),
        "status": "OK",
    }


def run_bronze_ingestion(
    raw_dir: Path | None = None,
    bronze_dir: Path | None = None,
) -> dict:
    """Top-level Bronze runner. Returns the manifest dict."""
    setup_logging()
    raw_dir = Path(raw_dir or config.RAW_INPUT_DIR)
    bronze_dir = Path(bronze_dir or config.BRONZE_DIR)
    bronze_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Bronze ingestion starting | raw=%s | bronze=%s", raw_dir, bronze_dir)

    entries = []
    for name, fname in config.SOURCE_FILES.items():
        entry = _ingest_one(name, raw_dir / fname, bronze_dir)
        entries.append(entry)

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "raw_dir": str(raw_dir),
        "bronze_dir": str(bronze_dir),
        "datasets": entries,
    }
    write_manifest(bronze_dir / "_manifest.json", manifest)

    missing = [e["dataset"] for e in entries if e["status"] != "OK"]
    if missing:
        logger.warning("Bronze finished with missing datasets: %s", missing)
    else:
        logger.info("Bronze ingestion complete (%d datasets)", len(entries))
    return manifest


if __name__ == "__main__":
    run_bronze_ingestion()
