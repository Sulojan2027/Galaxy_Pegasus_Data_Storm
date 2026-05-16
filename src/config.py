"""Project-wide configuration.

Centralizes paths, schema contracts, primary keys, and DQ thresholds so that
every layer of the pipeline reads from a single source of truth.

If the raw CSV column names differ from what is declared here, update the
`SCHEMAS` block below; the rest of the pipeline is column-name driven and
will adapt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
REJECTED_DIR = SILVER_DIR / "_rejected"
GOLD_DIR = DATA_DIR / "gold"
EXTERNAL_DIR = DATA_DIR / "external"
POI_CACHE_DIR = EXTERNAL_DIR / "poi"
PREDICTIONS_DIR = DATA_DIR / "predictions"
RAW_INPUT_DIR = DATA_DIR / "raw"  # where the user drops the original CSVs

for _p in (
    BRONZE_DIR,
    SILVER_DIR,
    REJECTED_DIR,
    GOLD_DIR,
    EXTERNAL_DIR,
    POI_CACHE_DIR,
    PREDICTIONS_DIR,
    RAW_INPUT_DIR,
):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source file mapping (rename if your CSVs are named differently)
# ---------------------------------------------------------------------------
SOURCE_FILES: dict[str, str] = {
    "transactions": "transactions_history_final.csv",
    "outlets": "outlet_master.csv",
    "seasonality": "distributor_seasonality_details.csv",
    "holidays": "holiday_list.csv",
}

# ---------------------------------------------------------------------------
# Schema contracts
# These are the *expected* canonical column names. Adjust the `aliases` dict
# in `normalize_columns()` (utils/io.py) if your raw files differ.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    primary_key: list[str]
    mandatory_cols: list[str]
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    date_cols: list[str] = field(default_factory=list)
    foreign_keys: dict[str, tuple[str, str]] = field(default_factory=dict)
    # foreign_keys: {local_col: (ref_dataset_name, ref_col)}


SCHEMAS: dict[str, DatasetSchema] = {
    "transactions": DatasetSchema(
        name="transactions",
        primary_key=["transaction_id"],
        mandatory_cols=["outlet_id", "distributor_id", "date", "volume_liters"],
        numeric_ranges={"volume_liters": (0.0, 50_000.0)},
        date_cols=["date"],
        foreign_keys={
            "outlet_id": ("outlets", "outlet_id"),
            "distributor_id": ("outlets", "distributor_id"),
        },
    ),
    "outlets": DatasetSchema(
        name="outlets",
        primary_key=["outlet_id"],
        mandatory_cols=["outlet_id", "distributor_id", "province"],
        numeric_ranges={
            "latitude": (5.5, 10.5),   # Sri Lanka bounding box
            "longitude": (79.0, 82.5),
        },
        date_cols=[],
    ),
    "seasonality": DatasetSchema(
        name="seasonality",
        primary_key=["distributor_id", "month"],
        mandatory_cols=["distributor_id", "month", "seasonality_index"],
        numeric_ranges={"seasonality_index": (0.0, 5.0), "month": (1, 12)},
        date_cols=[],
    ),
    "holidays": DatasetSchema(
        name="holidays",
        primary_key=["date", "holiday_name"],
        mandatory_cols=["date", "holiday_name"],
        numeric_ranges={},
        date_cols=["date"],
    ),
}

# ---------------------------------------------------------------------------
# Business / domain constants
# ---------------------------------------------------------------------------
TARGET_MONTH = "2026-01"        # The month we predict potential for
TARGET_MONTH_INT = 1            # January
EXPECTED_OUTLET_COUNT = 20_000  # Per the brief
EXPECTED_DISTRIBUTORS = [
    "DIST_W_01", "DIST_W_02", "DIST_W_03",
    "DIST_C_01", "DIST_C_02", "DIST_C_03",
    "DIST_NW_01", "DIST_NW_02",
    "DIST_S_01", "DIST_S_02",
]
EXPECTED_PROVINCES = ["Western", "Central", "North-Western", "Southern"]

# ---------------------------------------------------------------------------
# Data quality thresholds
# ---------------------------------------------------------------------------
DQ_CONFIG: dict[str, Any] = {
    "constant_run_min_days": 7,         # >=7 identical non-zero volumes flagged as suspicious
    "blackout_distributor_threshold": 0.95,  # if 95%+ of a distributor's outlets are zero on a day
    "round_number_suspicion_modulos": [50, 100, 500, 1000],  # credit-cap fingerprints
    "duplicate_strict": True,           # if True, all dupes after first are rejected
    "max_null_fraction": 0.30,          # if a mandatory col is >30% null we error out
}

# ---------------------------------------------------------------------------
# POI scraping
# ---------------------------------------------------------------------------
POI_CONFIG: dict[str, Any] = {
    "overpass_endpoint": "https://overpass-api.de/api/interpreter",
    "radii_meters": [500, 1000],
    "request_timeout_s": 60,
    "max_retries": 4,
    "backoff_base_s": 5,
    "batch_size": 25,                   # outlets per Overpass call
    # POI categories considered demand drivers for beverages.
    # Each entry is an Overpass tag filter snippet.
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

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
MODEL_CONFIG: dict[str, Any] = {
    "peer_cluster_count": 25,           # KMeans peer clusters
    "peer_ceiling_quantile": 0.90,      # use 90th percentile of peer best-months
    "quantile_regression_tau": 0.90,
    "ensemble_weights": {               # final weighted blend
        "peer_ceiling": 0.40,
        "quantile_regression": 0.30,
        "unconstrained_extrapolation": 0.30,
    },
    "min_unconstrained_months": 2,      # below this, fall back to peer ceiling only
    "potential_floor_multiplier": 1.0,  # potential must be >= historical max * this
}
