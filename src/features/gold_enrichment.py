"""Gold Layer — Feature Engineering / Enrichment."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.utils.io import read_parquet, setup_logging, write_parquet

logger = logging.getLogger(__name__)


# Loaders
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


# Monthly aggregation
def build_outlet_monthly(
    transactions: pd.DataFrame,
    seasonality: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate transactions to outlet × year × month."""
    if transactions.empty:
        return pd.DataFrame()

    df = transactions.copy()
    for c in ("year", "month", "volume_liters", "total_bill_value"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "_is_return" not in df.columns:
        df["_is_return"] = df["volume_liters"].lt(0).fillna(False)

    df["_returns_vol"] = np.where(df["_is_return"], -df["volume_liters"], 0.0)
    df["_purchase_vol"] = np.where(~df["_is_return"], df["volume_liters"], 0.0)
    df["_abs_vol"] = df["volume_liters"].abs()

    grouped = df.groupby(["outlet_id", "year", "month"], dropna=False)
    monthly = grouped.agg(
        volume_total=("volume_liters", "sum"),
        volume_gross=("_abs_vol", "sum"),
        volume_returns=("_returns_vol", "sum"),
        volume_purchases=("_purchase_vol", "sum"),
        bill_total=("total_bill_value", "sum"),
        sku_diversity=("sku_id", "nunique"),
        n_lines=("volume_liters", "size"),
        distributor_id=("distributor_id", "first"),
    ).reset_index()

    monthly["return_share"] = (monthly["volume_returns"] / monthly["volume_gross"].replace(0, np.nan)).fillna(0.0)

    # Constraint flagging at the outlet-MONTH level
    # 1. Stockout proxy
    outlet_median = monthly.groupby("outlet_id")["volume_total"].transform("median")
    monthly["low_volume_flag"] = (
        (monthly["volume_total"] > 0)
        & (monthly["volume_total"] < 0.5 * outlet_median)
    )

    # 2. Single-SKU month
    monthly["low_sku_diversity_flag"] = (
        monthly["sku_diversity"] < config.DQ_CONFIG["min_sku_diversity"]
    )

    # 3. Credit-cap fingerprint
    modulos = config.DQ_CONFIG["round_number_suspicion_modulos"]
    monthly["credit_cap_flag"] = False
    for m in modulos:
        is_round = (monthly["volume_total"].abs() > 0) & (monthly["volume_total"].abs() % m == 0)
        monthly["credit_cap_flag"] |= is_round

    # Unconstrained months reveal true demand.
    monthly["is_constrained"] = (
        monthly["low_volume_flag"]
        | monthly["low_sku_diversity_flag"]
        | monthly["credit_cap_flag"]
    )

    # Join numeric seasonality
    if not seasonality.empty:
        join_cols = [c for c in ("distributor_id", "year", "month")
                     if c in seasonality.columns and c in monthly.columns]
        if join_cols and "seasonality_index" in seasonality.columns:
            monthly = monthly.merge(
                seasonality[join_cols + ["seasonality_index"]].drop_duplicates(join_cols),
                on=join_cols, how="left",
            )
    if "seasonality_index" not in monthly.columns:
        monthly["seasonality_index"] = np.nan
    monthly["seasonality_index"] = monthly["seasonality_index"].fillna(1.0).replace(0, 1.0)
    monthly["volume_total_deflated"] = monthly["volume_total"] / monthly["seasonality_index"]

    return monthly


# Outlet-level rollup
def build_outlet_features(
    outlets: pd.DataFrame,
    monthly: pd.DataFrame,
    poi: pd.DataFrame,
) -> pd.DataFrame:
    """Outlet-level wide feature table consumed by the modeling layer."""

    if monthly.empty:
        logger.warning("monthly table empty — outlet features will be sparse")

    if not monthly.empty:
        unc = monthly[~monthly["is_constrained"]]
        agg = monthly.groupby("outlet_id").agg(
            months_observed=("month", "size"),
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
            hist_sku_diversity_mean=("sku_diversity", "mean"),
            hist_sku_diversity_max=("sku_diversity", "max"),
            hist_return_share_mean=("return_share", "mean"),
            hist_bill_total_mean=("bill_total", "mean"),
        ).reset_index()

        if not unc.empty:
            unc_agg = unc.groupby("outlet_id").agg(
                unc_deflated_max=("volume_total_deflated", "max"),
                unc_deflated_p90=("volume_total_deflated", lambda s: float(np.nanpercentile(s, 90))),
                unc_deflated_mean=("volume_total_deflated", "mean"),
                unc_months=("month", "size"),
            ).reset_index()
            agg = agg.merge(unc_agg, on="outlet_id", how="left")
    else:
        agg = pd.DataFrame(columns=["outlet_id"])

    df = outlets.merge(agg, on="outlet_id", how="left") if not outlets.empty else agg

    if not poi.empty and "outlet_id" in poi.columns:
        df = df.merge(poi, on="outlet_id", how="left")

    # One-hot small categorical attributes.
    if "outlet_size" in df.columns:
        df = df.join(pd.get_dummies(df["outlet_size"].astype("string"),
                                    prefix="size", dummy_na=False).astype(int))
    if "outlet_type" in df.columns:
        df = df.join(pd.get_dummies(df["outlet_type"].astype("string"),
                                    prefix="type", dummy_na=False).astype(int))
    if "province" in df.columns:
        df = df.join(pd.get_dummies(df["province"].astype("string"),
                                    prefix="prov", dummy_na=False).astype(int))

    if "hist_total_max" in df.columns and "hist_total_mean" in df.columns:
        df["volume_cv"] = (df["hist_total_std"] / df["hist_total_mean"]).replace(
            [np.inf, -np.inf], np.nan
        )
        df["max_to_mean_ratio"] = (df["hist_total_max"] / df["hist_total_mean"]).replace(
            [np.inf, -np.inf], np.nan
        )

    return df


# Runner
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
