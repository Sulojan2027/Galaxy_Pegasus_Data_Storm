"""Silver Layer — Cleaning, Quarantine, and Enrichment Joins.

Applies the DQ pipeline (from `src.processing.data_quality`) to each Bronze
dataset, then:

1. Writes clean rows to ``data/silver/<dataset>.parquet``
2. Writes rejected rows to ``data/silver/_rejected/<dataset>__<run_id>.parquet``
   with a `failure_reason` column.
3. Writes a per-dataset, per-check tally to ``data/silver/_summary.json``.

Beyond pure cleaning we also perform Silver-stage enrichment that every
downstream consumer needs:

- Map categorical `seasonality_index` → numeric using `SEASONALITY_NUMERIC`.
- Derive each outlet's primary `distributor_id` (from transactions) and
  attach it to the outlet master.
- Derive `province` from the distributor-ID prefix.
- Join the coordinate file into the outlet master.

Run as a script::

    python -m src.processing.silver_cleaning
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.processing.data_quality import CheckSpec, run_checks
from src.utils.io import read_parquet, setup_logging, write_parquet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-dataset check pipelines (declarative — easy to extend)
# ---------------------------------------------------------------------------
def _outlets_pipeline() -> list[CheckSpec]:
    s = config.SCHEMAS["outlets"]
    return [
        CheckSpec("null", {"mandatory_cols": s.mandatory_cols}),
        CheckSpec("duplicate", {"keys": s.primary_key}),
        # Normalize the dirty Outlet_Type / Outlet_Size string values.
        CheckSpec("text_normalize", {
            "col": "outlet_type",
            "mapping": config.OUTLET_TYPE_CANONICAL,
        }),
        CheckSpec("text_normalize", {
            "col": "outlet_size",
            "mapping": config.OUTLET_SIZE_CANONICAL,
        }),
        CheckSpec("value_range", {"col": "cooler_count", "min_": 0, "max_": 100}),
    ]


def _coordinates_pipeline() -> list[CheckSpec]:
    s = config.SCHEMAS["coordinates"]
    return [
        CheckSpec("null", {"mandatory_cols": s.mandatory_cols}),
        CheckSpec("duplicate", {"keys": s.primary_key}),
        # Attempt auto-fix for lat/lon swaps BEFORE bounding-box rejection.
        CheckSpec("coord_swap_fix", {
            "lat_col": "latitude",
            "lon_col": "longitude",
            "lat_bounds": s.numeric_ranges["latitude"],
            "lon_bounds": s.numeric_ranges["longitude"],
            "autofix": config.DQ_CONFIG["coord_swap_autofix"],
        }),
        CheckSpec("value_range", {
            "col": "latitude",
            "min_": s.numeric_ranges["latitude"][0],
            "max_": s.numeric_ranges["latitude"][1],
        }),
        CheckSpec("value_range", {
            "col": "longitude",
            "min_": s.numeric_ranges["longitude"][0],
            "max_": s.numeric_ranges["longitude"][1],
        }),
    ]


def _seasonality_pipeline() -> list[CheckSpec]:
    s = config.SCHEMAS["seasonality"]
    return [
        CheckSpec("null", {"mandatory_cols": s.mandatory_cols}),
        CheckSpec("duplicate", {"keys": s.primary_key}),
        CheckSpec("value_range", {"col": "month", "min_": 1, "max_": 12}),
        CheckSpec("value_range", {"col": "year", "min_": 2022, "max_": 2026}),
        # `seasonality_index` is categorical — accept only the 3 known values.
        CheckSpec("format", {
            "col": "seasonality_index",
            "regex": "|".join(config.SEASONALITY_NUMERIC.keys()),
        }),
    ]


def _holidays_pipeline() -> list[CheckSpec]:
    s = config.SCHEMAS["holidays"]
    return [
        CheckSpec("null", {"mandatory_cols": s.mandatory_cols}),
        CheckSpec("format", {"col": "date", "dtype": "datetime"}),
        CheckSpec("duplicate", {"keys": s.primary_key}),
    ]


def _transactions_pipeline(outlets_df: pd.DataFrame) -> list[CheckSpec]:
    s = config.SCHEMAS["transactions"]
    return [
        CheckSpec("null", {"mandatory_cols": s.mandatory_cols}),
        CheckSpec("format", {"col": "year", "dtype": "int"}),
        CheckSpec("format", {"col": "month", "dtype": "int"}),
        CheckSpec("value_range", {"col": "year", "min_": 2022, "max_": 2026}),
        CheckSpec("value_range", {"col": "month", "min_": 1, "max_": 12}),
        CheckSpec("referential_integrity", {
            "fk_col": "outlet_id",
            "ref_df": outlets_df,
            "ref_col": "outlet_id",
        }),
        # Tag-only: don't reject returns/credits.
        CheckSpec("negative_volume_tag", {"value_col": "volume_liters"}),
        # Reject exact duplicates on the natural composite key.
        CheckSpec("duplicate", {"keys": s.primary_key}),
        # NOTE: low-volume-month and credit-cap fingerprints are applied at
        # the MONTHLY-total grain inside `src/features/gold_enrichment.py`,
        # not at the transaction-line grain. Applying them per-line produced
        # huge false-positive rates (each line is a single SKU within a
        # month, so per-line quantiles are uninformative).
    ]


def _adjust_pipeline_to_columns(
    pipeline: list[CheckSpec], df: pd.DataFrame
) -> list[CheckSpec]:
    """Strip steps whose target columns aren't present (graceful degradation)."""
    keep: list[CheckSpec] = []
    for step in pipeline:
        params = step.params
        col = params.get("col")
        cols = (
            params.get("mandatory_cols")
            or params.get("keys")
            or params.get("group_keys")
        )
        if col and col not in df.columns:
            logger.info("skipping %s on missing col %s", step.name, col)
            continue
        if cols and not any(c in df.columns for c in cols):
            logger.info("skipping %s on missing cols %s", step.name, cols)
            continue
        keep.append(step)
    return keep


# ---------------------------------------------------------------------------
# Silver enrichment (after per-dataset cleaning)
# ---------------------------------------------------------------------------
def _enrich_seasonality(seasonality: pd.DataFrame) -> pd.DataFrame:
    """Map categorical seasonality_index → numeric float index."""
    if seasonality.empty or "seasonality_index" not in seasonality.columns:
        return seasonality
    out = seasonality.copy()
    out["seasonality_label"] = out["seasonality_index"].astype("string").str.strip()
    out["seasonality_index"] = out["seasonality_label"].map(config.SEASONALITY_NUMERIC).astype(float)
    return out


def _enrich_outlets(
    outlets: pd.DataFrame,
    coordinates: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Join coords + derive distributor_id and province per outlet."""
    if outlets.empty:
        return outlets

    df = outlets.copy()

    if not coordinates.empty:
        df = df.merge(
            coordinates[["outlet_id", "latitude", "longitude"]],
            on="outlet_id",
            how="left",
        )

    if not transactions.empty and "distributor_id" in transactions.columns:
        # Each outlet's transactions are confined to one distributor (verified
        # in EDA). Pick the most-frequent one defensively in case of stragglers.
        per_outlet_dist = (
            transactions.groupby("outlet_id")["distributor_id"]
            .agg(lambda s: s.value_counts().index[0])
            .rename("distributor_id")
            .reset_index()
        )
        df = df.merge(per_outlet_dist, on="outlet_id", how="left")

    if "distributor_id" in df.columns:
        df["province"] = df["distributor_id"].apply(_distributor_to_province)
    else:
        df["province"] = np.nan
    return df


def _distributor_to_province(dist_id: str | float) -> str | float:
    if not isinstance(dist_id, str):
        return np.nan
    for prefix, prov in config.DISTRIBUTOR_PREFIX_TO_PROVINCE.items():
        if dist_id.startswith(prefix):
            return prov
    return np.nan


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _process_one(
    name: str,
    pipeline: list[CheckSpec],
    bronze_path: Path,
    silver_dir: Path,
    rejected_dir: Path,
    run_id: str,
) -> dict:
    if not bronze_path.exists():
        logger.warning("bronze parquet missing for %s: %s", name, bronze_path)
        return {"dataset": name, "status": "MISSING"}

    df = read_parquet(bronze_path)
    pipeline = _adjust_pipeline_to_columns(pipeline, df)
    result = run_checks(df, name, pipeline)

    write_parquet(result.passing, silver_dir / f"{name}.parquet")

    if not result.rejected.empty:
        rej_path = rejected_dir / f"{name}__{run_id}.parquet"
        write_parquet(result.rejected, rej_path)

    return {
        "dataset": name,
        "status": "OK",
        "passing_rows": int(len(result.passing)),
        "rejected_rows": int(len(result.rejected)),
        "summary": result.summary,
    }


def run_silver_cleaning(
    bronze_dir: Path | None = None,
    silver_dir: Path | None = None,
    rejected_dir: Path | None = None,
) -> dict:
    setup_logging()
    bronze_dir = Path(bronze_dir or config.BRONZE_DIR)
    silver_dir = Path(silver_dir or config.SILVER_DIR)
    rejected_dir = Path(rejected_dir or config.REJECTED_DIR)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    # Step 1: clean each dataset independently.
    summaries: list[dict] = []

    summaries.append(_process_one(
        "outlets", _outlets_pipeline(),
        bronze_dir / "outlets.parquet", silver_dir, rejected_dir, run_id,
    ))
    summaries.append(_process_one(
        "coordinates", _coordinates_pipeline(),
        bronze_dir / "coordinates.parquet", silver_dir, rejected_dir, run_id,
    ))
    summaries.append(_process_one(
        "seasonality", _seasonality_pipeline(),
        bronze_dir / "seasonality.parquet", silver_dir, rejected_dir, run_id,
    ))
    summaries.append(_process_one(
        "holidays", _holidays_pipeline(),
        bronze_dir / "holidays.parquet", silver_dir, rejected_dir, run_id,
    ))

    # Reload clean outlets for the transactions RI check.
    clean_outlets_path = silver_dir / "outlets.parquet"
    clean_outlets = (
        read_parquet(clean_outlets_path)
        if clean_outlets_path.exists()
        else pd.DataFrame(columns=["outlet_id"])
    )

    summaries.append(_process_one(
        "transactions", _transactions_pipeline(clean_outlets),
        bronze_dir / "transactions.parquet", silver_dir, rejected_dir, run_id,
    ))

    # Step 2: Silver-stage enrichment & re-write enriched outlets/seasonality.
    transactions = (
        read_parquet(silver_dir / "transactions.parquet")
        if (silver_dir / "transactions.parquet").exists() else pd.DataFrame()
    )
    coordinates = (
        read_parquet(silver_dir / "coordinates.parquet")
        if (silver_dir / "coordinates.parquet").exists() else pd.DataFrame()
    )
    seasonality = (
        read_parquet(silver_dir / "seasonality.parquet")
        if (silver_dir / "seasonality.parquet").exists() else pd.DataFrame()
    )

    enriched_outlets = _enrich_outlets(clean_outlets, coordinates, transactions)
    write_parquet(enriched_outlets, silver_dir / "outlets.parquet")

    enriched_seasonality = _enrich_seasonality(seasonality)
    write_parquet(enriched_seasonality, silver_dir / "seasonality.parquet")

    summary_path = silver_dir / "_summary.json"
    summary_path.write_text(json.dumps({
        "run_id": run_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "datasets": summaries,
        "post_join": {
            "outlets_with_coords": int(enriched_outlets["latitude"].notna().sum())
                if "latitude" in enriched_outlets.columns else 0,
            "outlets_with_distributor": int(enriched_outlets["distributor_id"].notna().sum())
                if "distributor_id" in enriched_outlets.columns else 0,
            "outlets_total": int(len(enriched_outlets)),
        },
    }, indent=2, default=str))
    logger.info("Silver cleaning complete | summary -> %s", summary_path)
    return {"run_id": run_id, "datasets": summaries}


if __name__ == "__main__":
    run_silver_cleaning()
