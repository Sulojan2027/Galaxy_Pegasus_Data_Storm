"""Project-wide configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
REJECTED_DIR = SILVER_DIR / "_rejected"
GOLD_DIR = DATA_DIR / "gold"
EXTERNAL_DIR = DATA_DIR / "external"
POI_CACHE_DIR = EXTERNAL_DIR / "poi"
PREDICTIONS_DIR = DATA_DIR / "predictions"
RAW_INPUT_DIR = DATA_DIR / "raw"

for _p in (
    BRONZE_DIR, SILVER_DIR, REJECTED_DIR, GOLD_DIR,
    EXTERNAL_DIR, POI_CACHE_DIR, PREDICTIONS_DIR, RAW_INPUT_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)

# Source file mapping
SOURCE_FILES: dict[str, str] = {
    "transactions": "transactions_history_final.csv",
    "outlets":      "outlet_master.csv",
    "coordinates":  "outlet_coordinates.csv",
    "seasonality":  "distributor_seasonality_details.csv",
    "holidays":     "holiday_list.csv",
}

# Schema contracts


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    primary_key: list[str]
    mandatory_cols: list[str]
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    date_cols: list[str] = field(default_factory=list)
    foreign_keys: dict[str, tuple[str, str]] = field(default_factory=dict)


SCHEMAS: dict[str, DatasetSchema] = {
    "transactions": DatasetSchema(
        name="transactions",
        primary_key=["outlet_id", "year", "month", "distributor_id", "sku_id"],
        mandatory_cols=[
            "outlet_id", "year", "month", "distributor_id", "sku_id",
            "volume_liters", "total_bill_value",
        ],
        # Note: Volume_Liters can be negative (returns/credits).
        numeric_ranges={
            "year": (2022, 2026),
            "month": (1, 12),
        },
    ),
    "outlets": DatasetSchema(
        name="outlets",
        primary_key=["outlet_id"],
        mandatory_cols=["outlet_id", "outlet_type"],
        numeric_ranges={"cooler_count": (0, 100)},
    ),
    "coordinates": DatasetSchema(
        name="coordinates",
        primary_key=["outlet_id"],
        mandatory_cols=["outlet_id", "latitude", "longitude"],
        numeric_ranges={
            "latitude": (5.5, 10.5),    # Sri Lanka bounding box
            "longitude": (79.0, 82.5),
        },
    ),
    "seasonality": DatasetSchema(
        name="seasonality",
        primary_key=["distributor_id", "year", "month"],
        mandatory_cols=["distributor_id", "year", "month", "seasonality_index"],
        numeric_ranges={"year": (2022, 2026), "month": (1, 12)},
    ),
    "holidays": DatasetSchema(
        name="holidays",
        # Holiday can have multiple types; PK is the full triple.
        primary_key=["date", "holiday_name", "holiday_type"],
        mandatory_cols=["date", "holiday_name", "holiday_type"],
        date_cols=["date"],
    ),
}

# Business / domain constants
TARGET_MONTH_INT = 1            # January 2026
TARGET_MONTH_YEAR = 2026
TARGET_MONTH_LABEL = "2026-01"
SEASONALITY_FALLBACK_YEAR = 2025  # seasonality data covers 2023..2025; 2026 not present

EXPECTED_OUTLET_COUNT = 20_000
EXPECTED_DISTRIBUTORS = [
    "DIST_W_01", "DIST_W_02", "DIST_W_03",
    "DIST_C_01", "DIST_C_02", "DIST_C_03",
    "DIST_NW_01", "DIST_NW_02",
    "DIST_S_01", "DIST_S_02",
]
EXPECTED_PROVINCES = ["Western", "Central", "North-Western", "Southern"]
EXPECTED_SKUS = [f"SKU_{i:02d}" for i in range(1, 11)]

# Map distributor-ID prefix to province.
DISTRIBUTOR_PREFIX_TO_PROVINCE = {
    "DIST_W_": "Western",
    "DIST_C_": "Central",
    "DIST_NW_": "North-Western",
    "DIST_S_": "Southern",
}

# Categorical seasonality to multiplicative numeric index.
SEASONALITY_NUMERIC: dict[str, float] = {
    "Favorable":    1.15,
    "Moderate":     1.00,
    "Un-Favorable": 0.85,
}

# Outlet attribute normalization: maps observed dirty values to canonical ones.
OUTLET_TYPE_CANONICAL: dict[str, str] = {
    "grocry": "Grocery",
    "grocery": "Grocery",
    "bakry": "Bakery",
    "bakery": "Bakery",
    "eatery": "Eatery",
    "hotel": "Hotel",
    "pharmacy": "Pharmacy",
    "kiosk": "Kiosk",
    "smmt": "Supermarket",   # best-guess interpretation
}
OUTLET_SIZE_CANONICAL: dict[str, str] = {
    "small": "Small",
    "medium": "Medium",
    "large": "Large",
    "extra large": "Extra Large",
}

# Data quality thresholds
DQ_CONFIG: dict[str, Any] = {
    "round_number_suspicion_modulos": [50, 100, 500, 1000],
    "duplicate_strict": True,
    "max_null_fraction": 0.30,
    "coord_swap_autofix": True,    # try to swap lat/lon when one looks like the other
    "negative_volume_tag_only": True,  # don't reject returns
    "min_sku_diversity": 2,        # months with <2 SKUs flagged as constrained
    "low_volume_quantile": 0.20,   # months below 20th percentile (per outlet) flagged
    "near_max_share_threshold": 0.20,  # tightness near outlet's own max
}

# POI scraping
POI_CONFIG: dict[str, Any] = {
    "overpass_endpoint": "https://overpass-api.de/api/interpreter",
    "radii_meters": [500, 1000],
    "request_timeout_s": 60,
    "max_retries": 4,
    "backoff_base_s": 5,
    "batch_size": 25,
    "poi_taxonomy": {
        "school":      '["amenity"~"school|college|kindergarten"]',
        "university":  '["amenity"="university"]',
        "bus_stand":   '["highway"="bus_stop"]',
        "bus_station": '["amenity"="bus_station"]',
        "railway":     '["railway"="station"]',
        "hospital":    '["amenity"~"hospital|clinic"]',
        "place_of_worship": '["amenity"="place_of_worship"]',
        "tourism":     '["tourism"~"attraction|museum|viewpoint|hotel|guest_house"]',
        "market":      '["amenity"="marketplace"]',
        "government":  '["amenity"~"townhall|courthouse|police"]',
        "restaurant":  '["amenity"~"restaurant|cafe|fast_food"]',
        "sports":      '["leisure"~"sports_centre|stadium|pitch"]',
        "shop":        '["shop"]',
    },
}

# Modeling
MODEL_CONFIG: dict[str, Any] = {
    "peer_cluster_count": 25,
    "peer_ceiling_quantile": 0.90,
    "quantile_regression_tau": 0.90,
    "ensemble_weights": {
        "peer_ceiling": 0.40,
        "quantile_regression": 0.30,
        "unconstrained_extrapolation": 0.30,
    },
    "min_unconstrained_months": 2,
    "potential_floor_multiplier": 1.0,
}
