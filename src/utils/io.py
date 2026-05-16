"""I/O helpers used across every layer of the pipeline.

- Parquet preferred for intermediate storage (fast + typed).
- Column-name normalization centralized here so the rest of the code can
  reference canonical names regardless of how the raw CSV phrased them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column-name normalization
# ---------------------------------------------------------------------------
# Maps a wide range of plausible raw column names to the canonical names used
# by SCHEMAS in src/config.py. Extend freely as you encounter new aliases.
CANONICAL_ALIASES: dict[str, list[str]] = {
    "outlet_id":     ["outlet_id", "outletid", "store_id", "shop_id", "retailer_id"],
    "distributor_id": ["distributor_id", "distributorid", "dist_id", "distributor"],
    "date":          ["date", "txn_date", "transaction_date", "invoice_date", "order_date"],
    "year":          ["year", "yr", "transaction_year"],
    "month":         ["month", "mon", "month_num", "transaction_month"],
    "volume_liters": ["volume_liters", "volume", "qty_liters", "quantity_liters",
                      "litres", "liters", "qty", "volume_ltr"],
    "total_bill_value": ["total_bill_value", "bill_value", "total_value", "value",
                         "revenue", "total_revenue", "amount"],
    "sku_id":        ["sku_id", "sku", "product_id", "item_id"],
    "product_name":  ["product_name", "product", "item_name", "sku_name"],
    "outlet_size":   ["outlet_size", "outlet_class"],
    "cooler_count":  ["cooler_count", "coolers", "fridge_count", "num_coolers"],
    "outlet_type":   ["outlet_type", "channel"],
    "province":      ["province", "state", "region"],
    "district":      ["district", "area"],
    "latitude":      ["latitude", "lat", "gps_lat"],
    "longitude":     ["longitude", "lon", "lng", "gps_lon"],
    "seasonality_index": ["seasonality_index", "season_index",
                          "seasonality_factor", "seasonality_tag"],
    "holiday_name":  ["holiday_name", "holiday_description"],
    "holiday_type":  ["holiday_type"],
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names using `CANONICAL_ALIASES`.

    Unknown columns are kept as-is but slugged (lowercased, snake_cased) so the
    DataFrame remains addressable by simple strings.
    """
    # First, slug every column
    df = df.rename(columns=lambda c: _slug(str(c)))

    inverse_map: dict[str, str] = {}
    for canonical, aliases in CANONICAL_ALIASES.items():
        for alias in aliases:
            inverse_map[_slug(alias)] = canonical

    df = df.rename(columns=lambda c: inverse_map.get(c, c))
    return df


# ---------------------------------------------------------------------------
# Parquet / CSV helpers
# ---------------------------------------------------------------------------
def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("wrote %d rows -> %s", len(df), path)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info("read %d rows <- %s", len(df), path)
    return df


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("wrote %d rows -> %s", len(df), path)
    return path


def read_csv_resilient(path: Path) -> pd.DataFrame:
    """Read a CSV defensively (raw legacy exports are notoriously messy)."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode {path} with utf-8/latin-1")


# ---------------------------------------------------------------------------
# Manifest helpers (auditability)
# ---------------------------------------------------------------------------
def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(level: int = logging.INFO) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def safe_iter(it: Iterable | None) -> Iterable:
    return it if it is not None else []
