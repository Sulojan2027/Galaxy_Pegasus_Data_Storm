"""POI scraping via the OpenStreetMap Overpass API."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src import config
from src.utils.io import (
    read_parquet,
    setup_logging,
    write_parquet,
)

logger = logging.getLogger(__name__)

POI_RAW_DIR = config.POI_CACHE_DIR / "raw"
POI_RAW_DIR.mkdir(parents=True, exist_ok=True)


# Overpass query builder
def build_overpass_query(
    lat: float,
    lon: float,
    radius_m: int,
    taxonomy: dict[str, str],
    timeout_s: int = 60,
) -> str:
    """Build an Overpass QL query returning POIs within radius."""
    filters = []
    for cat, tag_filter in taxonomy.items():
        # `out center` returns a representative point for ways/relations.
        filters.append(f'  nwr{tag_filter}(around:{radius_m},{lat},{lon});')
    body = "\n".join(filters)
    return f"""[out:json][timeout:{timeout_s}];
(
{body}
);
out center tags;
"""


# HTTP layer
def _overpass_post(
    query: str,
    endpoint: str = config.POI_CONFIG["overpass_endpoint"],
    timeout_s: int = config.POI_CONFIG["request_timeout_s"],
    max_retries: int = config.POI_CONFIG["max_retries"],
    backoff_base_s: float = config.POI_CONFIG["backoff_base_s"],
) -> dict | None:
    """POST to Overpass with exponential backoff. Returns parsed JSON or None."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                endpoint,
                data={"data": query},
                timeout=timeout_s + 10,
                headers={"User-Agent": "DataStorm7-POI/1.0"},
            )
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 504, 502, 503):
                wait = backoff_base_s * (2 ** (attempt - 1))
                logger.warning(
                    "Overpass %s (attempt %d/%d), sleeping %.1fs",
                    r.status_code, attempt, max_retries, wait,
                )
                time.sleep(wait)
                continue
            logger.error("Overpass non-retryable %s: %s", r.status_code, r.text[:200])
            return None
        except requests.RequestException as e:
            last_exc = e
            wait = backoff_base_s * (2 ** (attempt - 1))
            logger.warning(
                "Overpass exception %s (attempt %d/%d), sleeping %.1fs",
                e, attempt, max_retries, wait,
            )
            time.sleep(wait)
    logger.error("Overpass gave up after %d attempts (%s)", max_retries, last_exc)
    return None


# Per-outlet scrape
def _cache_path(outlet_id: str, radius_m: int) -> Path:
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in str(outlet_id))
    return POI_RAW_DIR / f"{safe}__{radius_m}.json"


def fetch_outlet_pois(
    outlet_id: str,
    lat: float,
    lon: float,
    radius_m: int,
    taxonomy: dict[str, str],
    refresh: bool = False,
) -> dict | None:
    """Fetch (or load from cache) the Overpass response for one outlet."""
    cache = _cache_path(outlet_id, radius_m)
    if cache.exists() and not refresh:
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupt cache %s — refetching", cache)

    if not (math.isfinite(lat) and math.isfinite(lon)):
        logger.info("outlet %s has invalid coords (%s,%s) — skipping", outlet_id, lat, lon)
        return None

    query = build_overpass_query(lat, lon, radius_m, taxonomy)
    data = _overpass_post(query)
    if data is None:
        return None
    cache.write_text(json.dumps(data))
    return data


# Feature extraction
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _classify_element(tags: dict[str, str], taxonomy: dict[str, str]) -> str | None:
    """Cheap, deterministic classifier — first matching category wins."""
    # Pre-compute simple checks.
    am = tags.get("amenity", "")
    tour = tags.get("tourism", "")
    hwy = tags.get("highway", "")
    rail = tags.get("railway", "")
    leisure = tags.get("leisure", "")
    shop = tags.get("shop", "")

    if am in {"school", "college", "kindergarten"}:
        return "school"
    if am == "university":
        return "university"
    if hwy == "bus_stop":
        return "bus_stand"
    if am == "bus_station":
        return "bus_station"
    if rail == "station":
        return "railway"
    if am in {"hospital", "clinic"}:
        return "hospital"
    if am == "place_of_worship":
        return "place_of_worship"
    if tour in {"attraction", "museum", "viewpoint", "hotel", "guest_house"}:
        return "tourism"
    if am == "marketplace":
        return "market"
    if am in {"townhall", "courthouse", "police"}:
        return "government"
    if am in {"restaurant", "cafe", "fast_food"}:
        return "restaurant"
    if leisure in {"sports_centre", "stadium", "pitch"}:
        return "sports"
    if shop:
        return "shop"
    # Ignore elements not in taxonomy.
    if any(c in taxonomy for c in {am, tour, hwy, rail, leisure}):
        return None
    return None


def _features_from_response(
    outlet_id: str,
    lat: float,
    lon: float,
    radius_m: int,
    response: dict | None,
    taxonomy: dict[str, str],
) -> dict[str, Any]:
    feats: dict[str, Any] = {"outlet_id": outlet_id, "_radius_m": radius_m}
    categories = list(taxonomy.keys())
    counts = {c: 0 for c in categories}
    nearest = {c: float("inf") for c in categories}

    if response is None:
        for c in categories:
            feats[f"poi_count_{c}_{radius_m}m"] = 0
            feats[f"poi_nearest_{c}_{radius_m}m"] = float("nan")
        feats[f"poi_total_{radius_m}m"] = 0
        return feats

    for el in response.get("elements", []):
        tags = el.get("tags", {}) or {}
        cat = _classify_element(tags, taxonomy)
        if cat is None:
            continue
        counts[cat] = counts.get(cat, 0) + 1
        # Extract position depending on node type.
        elat = el.get("lat") or (el.get("center") or {}).get("lat")
        elon = el.get("lon") or (el.get("center") or {}).get("lon")
        if elat is not None and elon is not None:
            d = _haversine_m(lat, lon, elat, elon)
            if d < nearest[cat]:
                nearest[cat] = d

    total = 0
    for c in categories:
        feats[f"poi_count_{c}_{radius_m}m"] = counts[c]
        feats[f"poi_nearest_{c}_{radius_m}m"] = (
            float("nan") if not math.isfinite(nearest[c]) else float(nearest[c])
        )
        total += counts[c]
    feats[f"poi_total_{radius_m}m"] = total
    return feats


# Top-level builder
def build_poi_features(
    outlets_df: pd.DataFrame | None = None,
    radii: list[int] | None = None,
    taxonomy: dict[str, str] | None = None,
    refresh: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Run scraping over all outlets and return a wide POI feature table."""
    setup_logging()
    radii = radii or list(config.POI_CONFIG["radii_meters"])
    taxonomy = taxonomy or dict(config.POI_CONFIG["poi_taxonomy"])

    if outlets_df is None:
        path = config.SILVER_DIR / "outlets.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing silver outlets at {path}")
        outlets_df = read_parquet(path)

    needed = {"outlet_id", "latitude", "longitude"}
    if not needed.issubset(outlets_df.columns):
        raise ValueError(f"outlets_df must have {needed}; has {set(outlets_df.columns)}")

    df = outlets_df[["outlet_id", "latitude", "longitude"]].dropna().copy()
    if limit:
        df = df.head(limit)

    logger.info("scraping POIs for %d outlets × %d radii", len(df), len(radii))

    rows_by_outlet: dict[str, dict[str, Any]] = {}
    for i, r in df.iterrows():
        oid = str(r["outlet_id"])
        rows_by_outlet.setdefault(oid, {"outlet_id": oid})
        for radius in radii:
            resp = fetch_outlet_pois(
                oid, float(r["latitude"]), float(r["longitude"]),
                radius, taxonomy, refresh=refresh,
            )
            feats = _features_from_response(
                oid, float(r["latitude"]), float(r["longitude"]),
                radius, resp, taxonomy,
            )
            feats.pop("_radius_m", None)
            feats.pop("outlet_id", None)
            rows_by_outlet[oid].update(feats)
        if (i + 1) % 50 == 0:
            logger.info("progress: %d/%d outlets", i + 1, len(df))

    out = pd.DataFrame.from_records(list(rows_by_outlet.values()))
    out_path = config.POI_CACHE_DIR / "poi_features.parquet"
    write_parquet(out, out_path)
    logger.info("POI features ready: %s", out_path)
    return out


if __name__ == "__main__":
    build_poi_features()
