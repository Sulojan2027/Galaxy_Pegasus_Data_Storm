"""Latent Potential Estimation — three independent estimators + ensemble."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src import config
from src.utils.io import read_parquet, setup_logging, write_csv, write_parquet

logger = logging.getLogger(__name__)


# Feature selection helpers
def _select_numeric_features(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _prep_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0))
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    return sc.fit_transform(imp.fit_transform(df[cols]))


# 1. Peer ceiling
def estimate_peer_ceiling(
    features: pd.DataFrame,
    k: int | None = None,
    quantile: float | None = None,
) -> pd.Series:
    """Cluster outlets to find peer ceiling."""
    k = k or config.MODEL_CONFIG["peer_cluster_count"]
    quantile = quantile or config.MODEL_CONFIG["peer_ceiling_quantile"]
    df = features.copy()

    cluster_cols = [
        c for c in df.columns
        if c.startswith("poi_") or c in {"latitude", "longitude",
                                          "hist_total_p90", "hist_total_median"}
    ]
    X = _prep_matrix(df, cluster_cols)
    if X.shape[1] == 0 or len(df) < k:
        logger.warning("peer ceiling: insufficient features/rows — fallback to global")
        df["_peer_cluster"] = 0
    else:
        df["_peer_cluster"] = KMeans(
            n_clusters=min(k, len(df)), n_init=5, random_state=42
        ).fit_predict(X)

    ceiling_col = "unc_deflated_p90" if "unc_deflated_p90" in df.columns else "hist_deflated_p90"
    if ceiling_col not in df.columns:
        # last-ditch fallback
        df[ceiling_col] = df.get("hist_total_p90", np.nan)

    peer_q = df.groupby("_peer_cluster")[ceiling_col].quantile(quantile)
    df["_peer_q"] = df["_peer_cluster"].map(peer_q)

    # Rescale to each outlet's relative size within the cluster
    scale_col = "hist_total_median" if "hist_total_median" in df.columns else "hist_total_mean"
    if scale_col in df.columns:
        cluster_med = df.groupby("_peer_cluster")[scale_col].transform("median")
        ratio = (df[scale_col] / cluster_med.replace(0, np.nan)).clip(lower=0.5, upper=2.0).fillna(1.0)
        df["peer_ceiling"] = df["_peer_q"] * ratio
    else:
        df["peer_ceiling"] = df["_peer_q"]

    return df["peer_ceiling"]


# 2. Quantile regression
def estimate_quantile_regression(
    features: pd.DataFrame,
    target_col: str | None = None,
    tau: float | None = None,
) -> pd.Series:
    """Fit a high-quantile regression on outlet-level features."""
    tau = tau or config.MODEL_CONFIG["quantile_regression_tau"]

    target_col = target_col or (
        "hist_deflated_max" if "hist_deflated_max" in features.columns
        else "hist_total_max" if "hist_total_max" in features.columns
        else None
    )
    if target_col is None or features.empty:
        logger.warning("quantile regression: no target — returning NaNs")
        return pd.Series(np.nan, index=features.index)

    df = features.copy()
    feature_cols = _select_numeric_features(
        df,
        exclude={target_col, "outlet_id", "_peer_cluster", "peer_ceiling"}
        | {c for c in df.columns if c.startswith("hist_")  # avoid trivial target leakage
                   and c != "hist_total_median"},
    )
    if not feature_cols:
        logger.warning("quantile regression: no features — returning NaNs")
        return pd.Series(np.nan, index=features.index)

    X = _prep_matrix(df, feature_cols)
    y = pd.to_numeric(df[target_col], errors="coerce").values
    train_mask = ~np.isnan(y)

    if train_mask.sum() < max(50, X.shape[1] * 5):
        logger.warning(
            "quantile regression: insufficient training rows (%d) — returning NaNs",
            int(train_mask.sum()),
        )
        return pd.Series(np.nan, index=features.index)

    try:
        from sklearn.linear_model import QuantileRegressor
        qr = QuantileRegressor(quantile=tau, alpha=0.001, solver="highs")
        qr.fit(X[train_mask], y[train_mask])
        preds = qr.predict(X)
    except Exception as e:    # broad on purpose — fallback is critical for hackathon
        logger.warning("QuantileRegressor failed (%s) — falling back to GBR median", e)
        from sklearn.ensemble import GradientBoostingRegressor
        gbr = GradientBoostingRegressor(loss="quantile", alpha=tau, random_state=42)
        gbr.fit(X[train_mask], y[train_mask])
        preds = gbr.predict(X)

    preds = np.clip(preds, 0, None)
    return pd.Series(preds, index=features.index, name="quantile_regression")


# 3. Unconstrained extrapolation
def estimate_unconstrained_extrapolation(
    features: pd.DataFrame,
    seasonality: pd.DataFrame,
    target_month: int = config.TARGET_MONTH_INT,
    target_year: int = config.TARGET_MONTH_YEAR,
    fallback_year: int = config.SEASONALITY_FALLBACK_YEAR,
) -> pd.Series:
    """Extrapolate target month potential using unconstrained max."""
    df = features.copy()
    min_months = config.MODEL_CONFIG["min_unconstrained_months"]

    season_idx = pd.Series(1.0, index=df.index)
    if not seasonality.empty and "distributor_id" in df.columns:
        s = seasonality.copy()
        for c in ("seasonality_index", "month", "year"):
            if c in s.columns:
                s[c] = pd.to_numeric(s[c], errors="coerce")
        if "year" in s.columns:
            target_season = s[(s["month"] == target_month) & (s["year"] == target_year)]
            if target_season.empty:
                target_season = s[(s["month"] == target_month) & (s["year"] == fallback_year)]
        else:
            target_season = s[s["month"] == target_month]
        target_season = target_season[["distributor_id", "seasonality_index"]].drop_duplicates("distributor_id")
        mapping = dict(zip(target_season["distributor_id"], target_season["seasonality_index"]))
        season_idx = df["distributor_id"].map(mapping).fillna(1.0)
    df["_target_season"] = season_idx.replace(0, 1.0).fillna(1.0)

    base = df.get("unc_deflated_max")
    if base is None:
        logger.warning("no unconstrained max column — extrapolation falls back to global max")
        base = df.get("hist_deflated_max", pd.Series(np.nan, index=df.index))

    months_unc = df.get("months_unconstrained", pd.Series(0, index=df.index)).fillna(0)
    eligible = months_unc >= min_months

    out = pd.Series(np.nan, index=df.index, name="unconstrained_extrapolation")
    out.loc[eligible] = (base.loc[eligible] * df.loc[eligible, "_target_season"]).values
    return out


# Ensemble
def ensemble_predictions(
    peer: pd.Series,
    qreg: pd.Series,
    unc: pd.Series,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Weighted blend with graceful NaN handling."""
    weights = weights or config.MODEL_CONFIG["ensemble_weights"]
    methods = {
        "peer_ceiling": peer.reindex(peer.index),
        "quantile_regression": qreg.reindex(peer.index),
        "unconstrained_extrapolation": unc.reindex(peer.index),
    }
    stacked = pd.DataFrame(methods)
    w = pd.Series({k: weights.get(k, 0.0) for k in methods}, dtype=float)

    valid = stacked.notna() & (stacked >= 0)
    # renormalize weights per row over valid methods
    row_w = valid.astype(float).mul(w, axis=1)
    row_w_sum = row_w.sum(axis=1).replace(0, np.nan)
    blended = (stacked.fillna(0) * row_w).sum(axis=1) / row_w_sum
    return blended.rename("ensemble_potential")


# Top-level runner
def run_modeling(
    gold_dir: Path | None = None,
    silver_dir: Path | None = None,
    predictions_dir: Path | None = None,
    team_name: str = "galaxy_pegasus",
) -> dict[str, Any]:
    setup_logging()
    gold_dir = Path(gold_dir or config.GOLD_DIR)
    silver_dir = Path(silver_dir or config.SILVER_DIR)
    predictions_dir = Path(predictions_dir or config.PREDICTIONS_DIR)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    features_path = gold_dir / "outlet_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(f"missing gold features at {features_path}")
    features = read_parquet(features_path)

    season_path = silver_dir / "seasonality.parquet"
    seasonality = read_parquet(season_path) if season_path.exists() else pd.DataFrame()

    logger.info("modeling on %d outlets, %d features", len(features), features.shape[1])

    peer = estimate_peer_ceiling(features)
    qreg = estimate_quantile_regression(features)
    unc = estimate_unconstrained_extrapolation(features, seasonality)
    blended = ensemble_predictions(peer, qreg, unc)

    # Sanity floor: potential cannot be below historical max
    hist_max = features.get("hist_total_max", pd.Series(0, index=features.index)).fillna(0)
    floor_mult = config.MODEL_CONFIG["potential_floor_multiplier"]
    blended = np.maximum(blended.fillna(hist_max * floor_mult), hist_max * floor_mult)

    out = features[["outlet_id"]].copy()
    out["peer_ceiling"] = peer.values
    out["quantile_regression"] = qreg.values
    out["unconstrained_extrapolation"] = unc.values
    out["Maximum_Monthly_Liters"] = blended.values

    # Audit / diagnostics
    diag_path = predictions_dir / "modeling_diagnostics.parquet"
    write_parquet(out, diag_path)

    # Final deliverable CSV
    final = out[["outlet_id", "Maximum_Monthly_Liters"]].rename(
        columns={"outlet_id": "Outlet_ID"}
    )
    final["Maximum_Monthly_Liters"] = final["Maximum_Monthly_Liters"].round(2)
    final_path = predictions_dir / f"{team_name}_predictions.csv"
    write_csv(final, final_path)

    logger.info("predictions written -> %s", final_path)
    return {
        "predictions_csv": final_path,
        "diagnostics_parquet": diag_path,
        "n_outlets": int(len(final)),
        "coverage": {
            "peer_ceiling": int(peer.notna().sum()),
            "quantile_regression": int(qreg.notna().sum()),
            "unconstrained_extrapolation": int(unc.notna().sum()),
        },
    }


if __name__ == "__main__":
    run_modeling()
