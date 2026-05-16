"""Silver Layer — Cleaning & Quarantine.

Applies the DQ pipeline (from `src.processing.data_quality`) to each Bronze
dataset, then writes:

- ``data/silver/<dataset>.parquet``     — clean rows
- ``data/silver/_rejected/<dataset>__<run_id>.parquet`` — rejected rows w/ reasons
- ``data/silver/_summary.json``         — per-dataset, per-check tallies

Run as a script::

    python -m src.processing.silver_cleaning
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

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
        CheckSpec("format", {"col": "outlet_id", "dtype": "string"}),
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
        CheckSpec("value_range", {
            "col": "seasonality_index",
            "min_": s.numeric_ranges["seasonality_index"][0],
            "max_": s.numeric_ranges["seasonality_index"][1],
        }),
    ]


def _holidays_pipeline() -> list[CheckSpec]:
    s = config.SCHEMAS["holidays"]
    return [
        CheckSpec("null", {"mandatory_cols": s.mandatory_cols}),
        CheckSpec("format", {"col": "date", "dtype": "date"}),
        CheckSpec("duplicate", {"keys": s.primary_key}),
    ]


def _transactions_pipeline(outlets_df: pd.DataFrame) -> list[CheckSpec]:
    s = config.SCHEMAS["transactions"]
    return [
        CheckSpec("null", {"mandatory_cols": s.mandatory_cols}),
        CheckSpec("format", {"col": "date", "dtype": "date"}),
        CheckSpec("value_range", {
            "col": "volume_liters",
            "min_": s.numeric_ranges["volume_liters"][0],
            "max_": s.numeric_ranges["volume_liters"][1],
        }),
        CheckSpec("referential_integrity", {
            "fk_col": "outlet_id",
            "ref_df": outlets_df,
            "ref_col": "outlet_id",
        }),
        # If transaction_id exists, use it; otherwise dedupe on a composite key.
        CheckSpec("duplicate", {
            "keys": ["transaction_id"] if "transaction_id" in outlets_df.columns
                    else ["outlet_id", "date", "volume_liters"],
        }),
        CheckSpec("constant_run", {
            "group_keys": ["outlet_id"],
            "order_col": "date",
            "value_col": "volume_liters",
            "min_run": config.DQ_CONFIG["constant_run_min_days"],
        }),
        CheckSpec("distributor_blackout", {
            "distributor_col": "distributor_id",
            "date_col": "date",
            "outlet_col": "outlet_id",
            "value_col": "volume_liters",
            "blackout_threshold": config.DQ_CONFIG["blackout_distributor_threshold"],
        }),
        # credit-cap is TAG-only (doesn't reject); kept last so it sees clean rows.
        CheckSpec("credit_cap_signature", {
            "group_keys": ["outlet_id"],
            "value_col": "volume_liters",
            "modulos": config.DQ_CONFIG["round_number_suspicion_modulos"],
        }),
    ]


def _adjust_pipeline_to_columns(
    pipeline: list[CheckSpec], df: pd.DataFrame
) -> list[CheckSpec]:
    """Strip steps whose target columns aren't present (graceful degradation)."""
    keep: list[CheckSpec] = []
    for step in pipeline:
        params = step.params
        col = params.get("col")
        cols = params.get("mandatory_cols") or params.get("keys")
        # null/duplicate keep at least one valid col
        if col and col not in df.columns:
            logger.info("skipping %s on missing col %s", step.name, col)
            continue
        if cols and not any(c in df.columns for c in cols):
            logger.info("skipping %s on missing cols %s", step.name, cols)
            continue
        keep.append(step)
    return keep


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

    # We need outlets first so transactions can do RI against it.
    outlets_bronze = bronze_dir / "outlets.parquet"
    outlets_df = (
        read_parquet(outlets_bronze)
        if outlets_bronze.exists() else pd.DataFrame(columns=["outlet_id"])
    )

    summaries: list[dict] = []
    summaries.append(_process_one(
        "outlets", _outlets_pipeline(),
        bronze_dir / "outlets.parquet", silver_dir, rejected_dir, run_id,
    ))
    # reload the clean outlets for downstream RI
    clean_outlets_path = silver_dir / "outlets.parquet"
    clean_outlets = (
        read_parquet(clean_outlets_path)
        if clean_outlets_path.exists() else outlets_df
    )

    summaries.append(_process_one(
        "seasonality", _seasonality_pipeline(),
        bronze_dir / "seasonality.parquet", silver_dir, rejected_dir, run_id,
    ))
    summaries.append(_process_one(
        "holidays", _holidays_pipeline(),
        bronze_dir / "holidays.parquet", silver_dir, rejected_dir, run_id,
    ))
    summaries.append(_process_one(
        "transactions", _transactions_pipeline(clean_outlets),
        bronze_dir / "transactions.parquet", silver_dir, rejected_dir, run_id,
    ))

    summary_path = silver_dir / "_summary.json"
    summary_path.write_text(json.dumps({
        "run_id": run_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "datasets": summaries,
    }, indent=2, default=str))
    logger.info("Silver cleaning complete | summary -> %s", summary_path)
    return {"run_id": run_id, "datasets": summaries}


if __name__ == "__main__":
    run_silver_cleaning()
