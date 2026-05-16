"""Gold Layer — Feature Engineering / Enrichment.

Builds the model-ready table at the **outlet level**, combining:

- Cleaned transactions (Silver)         — observed volume signals
- Outlet master (Silver)                — geo + channel attributes
- Distributor seasonality (Silver)      — month indices
- Holidays (Silver)                     — activity context
- POI features (external/poi)           — catchment / footfall proxies

Key outputs:

- ``data/gold/outlet_monthly.parquet`` — long-format outlet × month
  observations with `is_constrained` flag (for the modeling layer).
- ``data/gold/outlet_features.parquet`` — wide outlet-level feature table
  used by the peer-ceiling, quantile-regression, and unconstrained-
  extrapolation estimators.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.utils.io import read_parquet, setup_logging, write_parquet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _safe_read(name: str) -> pd.DataFrame:
    p = config.SILVER_DIR / f"{name}.parquet"
    if not p.exists():
        logger.warning("Silver dataset missing: %s", p)
        return pd.DataFrame()
    return read_parquet(p)


def _load_poi() -> pd.DataFrame:
    p = config.POI_CACHE_DIR / "poi_features.parquet"
    if not p.exists():
        logger.info("POI features file not found at %s — skipping POI join", p)
        return pd.DataFrame(columns=["outlet_id"])
    return read_parquet(p)


# ---------------------------------------------------------------------------
# Monthly aggregation + constraint flagging
# ---------------------------------------------------------------------------
def build_outlet_monthly(
    transactions: pd.DataFrame,
    seasonality: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate transactions to outlet × month + flag constrained months.

    A month is flagged `is_constrained` if ANY of the following hold:
      - the outlet's credit-cap signature triggered for that month
      - the month had a stockout/blackout proxy (>=25% of days had zero volume
        despite the outlet being otherwise active)
      - the outlet hit its observed monthly max within ±2% multiple times
        (a soft ceiling fingerprint)

    Constrained months are NOT used as ground-truth ceilings downstream.
    """
    if transactions.empty:
        return pd.DataFrame()

    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "outlet_id"])
    df["volume_liters"] = pd.to_numeric(df["volume_liters"], errors="coerce").fillna(0.0)
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["month"] = df["date"].dt.month

    credit_cap_flag = (
        df["_credit_cap_flag"]
        if "_credit_cap_flag" in df.columns
        else pd.Series(False, index=df.index)
    )
    df["_credit_cap_flag"] = credit_cap_flag.fillna(False)

    grouped = df.groupby(["outlet_id", "year_month"], dropna=False)
    monthly = grouped.agg(
        volume_total=("volume_liters", "sum"),
        volume_max_day=("volume_liters", "max"),
        volume_mean_day=("volume_liters", "mean"),
        volume_p90_day=("volume_liters", lambda s: float(np.nanpercentile(s, 90))),
        active_days=("volume_liters", lambda s: int((s > 0).sum())),
        total_days=("volume_liters", "size"),
        any_credit_cap=("_credit_cap_flag", "any"),
        distributor_id=("distributor_id", "first") if "distributor_id" in df.columns else ("outlet_id", "first"),
        month=("month", "first"),
    ).reset_index()

    monthly["zero_day_share"] = 1.0 - (monthly["active_days"] / monthly["total_days"].clip(lower=1))
    monthly["stockout_flag"] = monthly["zero_day_share"] >= 0.25

    # Soft ceiling fingerprint: many days at near-max volume within the month
    def near_max_share(s: pd.Series) -> float:
        if s.empty or s.max() <= 0:
            return 0.0
        m = s.max()
        return float(((s >= 0.98 * m) & (s > 0)).mean())

    near_max = grouped["volume_liters"].apply(near_max_share).rename("near_max_share").reset_index()
    monthly = monthly.merge(near_max, on=["outlet_id", "year_month"], how="left")
    monthly["soft_ceiling_flag"] = monthly["near_max_share"] >= 0.40

    monthly["is_constrained"] = (
        monthly["any_credit_cap"].fillna(False)
        | monthly["stockout_flag"]
        | monthly["soft_ceiling_flag"]
    )

    # Seasonality-deflate the monthly totals so they're comparable across months.
    if not seasonality.empty:
        s = seasonality.copy()
        for c in ("seasonality_index", "month"):
            if c in s.columns:
                s[c] = pd.to_numeric(s[c], errors="coerce")
        join_keys = [c for c in ("distributor_id", "month") if c in s.columns and c in monthly.columns]
        if join_keys:
            monthly = monthly.merge(
                s[join_keys + ["seasonality_index"]].drop_duplicates(join_keys),
                on=join_keys, how="left",
            )
        else:
            monthly["seasonality_index"] = np.nan
    else:
        monthly["seasonality_index"] = np.nan
    monthly["seasonality_index"] = monthly["seasonality_index"].fillna(1.0).replace(0, 1.0)
    monthly["volume_total_deflated"] = monthly["volume_total"] / monthly["seasonality_index"]

    return monthly


# ---------------------------------------------------------------------------
# Outlet-level rollup
# ---------------------------------------------------------------------------
def build_outlet_features(
    outlets: pd.DataFrame,
    monthly: pd.DataFrame,
    poi: pd.DataFrame,
) -> pd.DataFrame:
    """Outlet-level wide feature table for the modeling layer."""

    if monthly.empty:
        logger.warning("monthly table empty — outlet features will be sparse")

    if not monthly.empty:
        unc = monthly[~monthly["is_constrained"]]
        agg = monthly.groupby("outlet_id").agg(
            months_observed=("year_month", "nunique"),
            months_constrained=("is_constrained", "sum"),
            months_unconstrained=("is_constrained", lambda s: int((~s).sum())),
            hist_total_mean=("volume_total", "mean"),
            hist_total_median=("volume_total", "median"),
            hist_total_p75=("volume_total", lambda s: float(np.nanpercentile(s, 75))),
            hist_total_p90=("volume_total", lambda s: float(np.nanpercentile(s, 90))),
            hist_total_p95=("volume_total", lambda s: float(np.nanpercentile(s, 95))),
            hist_total_max=("volume_total", "max"),
            hist_total_std=("volume_total", "std"),
            hist_deflated_max=("volume_total_deflated", "max"),
            hist_deflated_p90=("volume_total_deflated", lambda s: float(np.nanpercentile(s, 90))),
        ).reset_index()

        unc_agg = unc.groupby("outlet_id").agg(
            unc_deflated_max=("volume_total_deflated", "max"),
            unc_deflated_p90=("volume_total_deflated", lambda s: float(np.nanpercentile(s, 90))),
            unc_deflated_mean=("volume_total_deflated", "mean"),
            unc_months=("year_month", "nunique"),
        ).reset_index()
        agg = agg.merge(unc_agg, on="outlet_id", how="left")
    else:
        agg = pd.DataFrame(columns=["outlet_id"])

    df = outlets.merge(agg, on="outlet_id", how="left") if not outlets.empty else agg

    if not poi.empty and "outlet_id" in poi.columns:
        df = df.merge(poi, on="outlet_id", how="left")

    # Lightweight derived features
    if "hist_total_max" in df.columns and "hist_total_mean" in df.columns:
        df["volume_cv"] = (df["hist_total_std"] / df["hist_total_mean"]).replace([np.inf, -np.inf], np.nan)
        df["max_to_mean_ratio"] = (df["hist_total_max"] / df["hist_total_mean"]).replace([np.inf, -np.inf], np.nan)

    return df


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_gold_enrichment(
    silver_dir: Path | None = None,
    gold_dir: Path | None = None,
) -> dict[str, Path]:
    setup_logging()
    silver_dir = Path(silver_dir or config.SILVER_DIR)
    gold_dir = Path(gold_dir or config.GOLD_DIR)
    gold_dir.mkdir(parents=True, exist_ok=True)

    transactions = _safe_read("transactions")
    outlets = _safe_read("outlets")
    seasonality = _safe_read("seasonality")
    poi = _load_poi()

    monthly = build_outlet_monthly(transactions, seasonality)
    monthly_path = gold_dir / "outlet_monthly.parquet"
    write_parquet(monthly, monthly_path)

    features = build_outlet_features(outlets, monthly, poi)
    features_path = gold_dir / "outlet_features.parquet"
    write_parquet(features, features_path)

    logger.info("Gold enrichment complete: %s, %s", monthly_path, features_path)
    return {"monthly": monthly_path, "features": features_path}


if __name__ == "__main__":
    run_gold_enrichment()
