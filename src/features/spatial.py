"""Spatial enrichment — gravity accessibility + competitive saturation.

This module turns the raw distance-decayed POI signals produced by
``poi_scraper`` (component ①) and the geometry of our own outlet network into
two interpretable quantities, then combines them into a single bounded
``spatial_multiplier`` that the transparent model multiplies straight into the
prediction.

Two ideas, both framed as a Huff/gravity catchment model:

1. **Accessibility** (pull).  A gravity sum over surrounding demand-generating
   POIs, each weighted by (a) its type's Huff attractiveness ``A_j`` and
   (b) a distance-decay kernel already applied per POI type in the scraper.
   High accessibility ⇒ more footfall past the outlet ⇒ upward adjustment.

2. **Saturation** (competition / demand-splitting).  How many *other sellers*
   share this outlet's catchment — our own neighbouring outlets (from the
   coordinate file, via a BallTree haversine query) plus third-party OSM shops.
   High saturation ⇒ catchment demand is split ⇒ downward adjustment.

The multiplier is centred at 1.0 by construction (robust median/IQR
standardisation) and hard-clamped to ``SPATIAL_CONFIG[clamp_min, clamp_max]``
so a product of factors can never blow up:

    raw = access_beta * z(log accessibility) - sat_beta * z(log saturation)
    spatial_multiplier = clip(1.0 + raw, clamp_min, clamp_max)

Every intermediate (accessibility, own-network density, saturation) is returned
as a column so the XAI / narrative layer can read the decomposition directly.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

_EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------------------------
# Robust standardisation (median / IQR) — keeps the multiplier centred at 1.0
# ---------------------------------------------------------------------------
def _robust_z(s: pd.Series) -> pd.Series:
    """Median/IQR z-score. Robust to the long right tail of POI/competition
    counts, and exactly zero at the median so ``1.0 + beta*z`` is centred at 1.0.
    """
    x = pd.to_numeric(s, errors="coerce").astype(float)
    med = x.median()
    q75, q25 = x.quantile(0.75), x.quantile(0.25)
    iqr = q75 - q25
    # Convert IQR to a std-equivalent scale (1.349 ≈ IQR/σ for a normal).
    scale = iqr / 1.349 if iqr > 0 else (x.std(ddof=0) or 1.0)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return ((x - med) / scale).fillna(0.0)


# ---------------------------------------------------------------------------
# 1. Own-network density (competition we control)
# ---------------------------------------------------------------------------
def compute_own_network_density(
    features: pd.DataFrame,
    radius_m: float | None = None,
) -> pd.Series:
    """Count, for each outlet, how many OTHER of our outlets fall within
    ``radius_m`` metres. Uses a BallTree haversine query (O(n log n)).

    Returns a Series aligned to ``features.index`` (self is excluded).
    """
    radius_m = radius_m if radius_m is not None else config.SATURATION_CONFIG["own_radius_m"]
    out = pd.Series(0.0, index=features.index, name="own_network_density")

    if not {"latitude", "longitude"}.issubset(features.columns):
        logger.warning("own-network density: missing lat/lon — returning zeros")
        return out

    coords = features[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
    valid = coords.notna().all(axis=1)
    if valid.sum() == 0:
        return out

    try:
        from sklearn.neighbors import BallTree
    except Exception as e:  # pragma: no cover - sklearn is a hard dep elsewhere
        logger.warning("BallTree unavailable (%s) — own-network density = 0", e)
        return out

    rad = np.deg2rad(coords.loc[valid].to_numpy())
    tree = BallTree(rad, metric="haversine")
    r = radius_m / _EARTH_RADIUS_M  # angular radius for haversine metric
    counts = tree.query_radius(rad, r=r, count_only=True)
    # query_radius counts the point itself — subtract it.
    out.loc[valid] = (counts - 1).clip(min=0).astype(float)
    return out


# ---------------------------------------------------------------------------
# 2. Gravity accessibility (pull)
# ---------------------------------------------------------------------------
def _max_radius() -> int:
    return int(max(config.POI_CONFIG["radii_meters"]))


def compute_accessibility(features: pd.DataFrame, radius_m: int | None = None) -> pd.Series:
    """Huff accessibility: demand-weighted sum of per-type decay scores.

    Reads the ``poi_access_<cat>_<r>m`` columns emitted by the scraper at the
    widest scraped radius (the decay kernel already handles distance falloff,
    so the widest radius is the most complete catchment). ``shop`` is excluded
    here — it is competition, accounted for in saturation, not demand.
    """
    radius_m = radius_m or _max_radius()
    weights = config.POI_DECAY_CONFIG["demand_weight"]

    score = pd.Series(0.0, index=features.index, name="poi_accessibility")
    used = []
    for cat, w in weights.items():
        if w <= 0:
            continue
        col = f"poi_access_{cat}_{radius_m}m"
        if col in features.columns:
            score = score + w * pd.to_numeric(features[col], errors="coerce").fillna(0.0)
            used.append(cat)
    if not used:
        logger.warning("accessibility: no poi_access_*_%dm columns found — score=0", radius_m)
    return score


# ---------------------------------------------------------------------------
# 3. Saturation index (own network + OSM shops)
# ---------------------------------------------------------------------------
def compute_saturation(features: pd.DataFrame, own_density: pd.Series) -> pd.Series:
    """saturation = own_neighbours + osm_shop_weight * osm_shop_count.

    OSM shop count is read from the configured radius's count column.
    """
    cfg = config.SATURATION_CONFIG
    shop_col = f"poi_count_shop_{cfg['osm_shop_radius_m']}m"
    osm_shops = (
        pd.to_numeric(features[shop_col], errors="coerce").fillna(0.0)
        if shop_col in features.columns
        else pd.Series(0.0, index=features.index)
    )
    if shop_col not in features.columns:
        logger.warning("saturation: %s missing — using own-network only", shop_col)
    sat = own_density.fillna(0.0) + cfg["osm_shop_weight"] * osm_shops
    return sat.rename("saturation_index")


# ---------------------------------------------------------------------------
# 4. Bounded spatial multiplier
# ---------------------------------------------------------------------------
def compute_spatial_multiplier(
    accessibility: pd.Series,
    saturation: pd.Series,
) -> pd.Series:
    """Combine accessibility (lift) and saturation (discount) into a multiplier
    centred at 1.0 and clamped to ``SPATIAL_CONFIG[clamp_min, clamp_max]``.
    """
    cfg = config.SPATIAL_CONFIG
    # log1p compresses the heavy right tails before standardising.
    z_acc = _robust_z(np.log1p(accessibility.clip(lower=0)))
    z_sat = _robust_z(np.log1p(saturation.clip(lower=0)))
    raw = cfg["access_beta"] * z_acc - cfg["sat_beta"] * z_sat
    # Recenter on the median so the multiplier is a *relative* adjustment around
    # the typical outlet: the median outlet gets exactly 1.0, accessibility above
    # / saturation below the median lift it, the reverse discounts it. Without
    # this, the median of (access_beta*z_acc - sat_beta*z_sat) is not zero (the
    # median of a difference != difference of medians), so the factor acts as a
    # systematic inflator rather than a centered adjustment.
    raw = raw - raw.median()
    mult = (1.0 + raw).clip(lower=cfg["clamp_min"], upper=cfg["clamp_max"])
    return mult.rename("spatial_multiplier")


# ---------------------------------------------------------------------------
# Orchestrator — attach all spatial columns to an outlet-level frame
# ---------------------------------------------------------------------------
def attach_spatial_factors(features: pd.DataFrame) -> pd.DataFrame:
    """Return ``features`` with the spatial columns added:

    - ``own_network_density`` — our outlets within the own-network radius
    - ``poi_accessibility``   — Huff gravity pull score
    - ``saturation_index``    — competition density (own + OSM)
    - ``spatial_multiplier``  — bounded [clamp_min, clamp_max] adjustment factor
    """
    df = features.copy()
    df["own_network_density"] = compute_own_network_density(df)
    df["poi_accessibility"] = compute_accessibility(df)
    df["saturation_index"] = compute_saturation(df, df["own_network_density"])
    df["spatial_multiplier"] = compute_spatial_multiplier(
        df["poi_accessibility"], df["saturation_index"]
    )
    logger.info(
        "spatial factors | accessibility med=%.3f | saturation med=%.3f | "
        "spatial_multiplier: min=%.3f p50=%.3f max=%.3f",
        df["poi_accessibility"].median(),
        df["saturation_index"].median(),
        df["spatial_multiplier"].min(),
        df["spatial_multiplier"].median(),
        df["spatial_multiplier"].max(),
    )
    return df
