# Galaxy Pegasus — Data Storm v7.0 (Preliminary Round)

Team codebase for the **Data Storm 7.0 — Storming Round**, organized by the
Rotaract Club of University of Moratuwa and powered by OCTAVE — John Keells
Group.

The challenge: estimate the **Maximum Monthly Purchase Potential (in liters)**
for ~20,000 traditional-trade outlets across 10 distributors and 4 provinces
in Sri Lanka, for **January 2026**. There is no labeled target — observed
volume is left-censored by credit limits, stockouts, and capacity caps. Our
job is to *uncap* it.

## Lakehouse architecture

```
data/
├── raw/         <-- drop the 4 provided CSVs here (untracked)
├── bronze/      <-- raw → parquet, no transformations
├── silver/      <-- cleaned data after DQ engine
│   └── _rejected/   <-- quarantined rows with failure reasons
├── gold/        <-- model-ready feature tables
├── external/poi/<-- scraped OSM POI features (cached)
└── predictions/ <-- final CSV deliverable
```

The code under `src/` mirrors this:

| Layer  | Module                                            | Purpose                              |
|--------|---------------------------------------------------|--------------------------------------|
| Bronze | `src/ingestion/bronze_ingestion.py`               | Raw CSV → parquet + manifest         |
| DQ     | `src/processing/data_quality.py`                  | Reusable, parameterizable check engine|
| Silver | `src/processing/silver_cleaning.py`               | Apply DQ pipelines, write quarantine |
| POI    | `src/features/poi_scraper.py`                     | Overpass API w/ caching + retries    |
| Gold   | `src/features/gold_enrichment.py`                 | Outlet × month + outlet features     |
| Model  | `src/modeling/latent_potential_model.py`          | Triangulated potential estimator     |

## Quick start

```bash
# 1. Install dependencies (Python 3.11+ recommended)
pip install -r requirements.txt

# 2. Drop the four provided CSVs into data/raw/
#    Expected filenames (override in src/config.py if different):
#       transactions_history_final.csv
#       outlet_master.csv
#       distributor_seasonality_details.csv
#       holiday_list.csv

# 3. Run end-to-end
python run_pipeline.py

# Useful flags during development
python run_pipeline.py --skip-poi              # reuse cached POI features
python run_pipeline.py --poi-limit 100         # only scrape 100 outlets
python run_pipeline.py --refresh-poi           # ignore cache, re-scrape
```

Output is written to `data/predictions/galaxy_pegasus_predictions.csv`
with columns `Outlet_ID, Maximum_Monthly_Liters`.

## Running individual stages

```bash
python -m src.ingestion.bronze_ingestion
python -m src.processing.silver_cleaning
python -m src.features.poi_scraper
python -m src.features.gold_enrichment
python -m src.modeling.latent_potential_model
```

## Notebooks

All exploratory work lives in `notebooks/`, executed in order:

1. `01_data_forensics_eda.ipynb` — anomaly hunting, DQ findings, plots
2. `02_poi_scraping_exploration.ipynb` — OSM/Overpass exploration & sanity
3. `03_feature_engineering.ipynb` — Gold feature audits
4. `04_latent_potential_modeling.ipynb` — modeling, sensitivity analysis

## DQ engine — reusability contract

Every check function in `src/processing/data_quality.py` has the signature:

```python
def check_<name>(df: DataFrame, **params) -> tuple[DataFrame, DataFrame]:
    """Returns (passing_rows, rejected_rows_with_failure_reason)."""
```

Checks are registered in a dispatch registry; a **declarative pipeline** of
`CheckSpec(name, params)` is then applied to each dataset by
`run_checks(df, dataset_name, pipeline)`. Adding a new check is one function
+ one config entry.

Built-in checks:

| Check                    | Purpose                                                 |
|--------------------------|---------------------------------------------------------|
| `duplicate`              | Configurable composite-key dedupe                        |
| `null`                   | Mandatory-field null / empty-string detector             |
| `referential_integrity`  | FK ↔ PK validation across datasets                       |
| `value_range`            | Numeric min/max boundary                                 |
| `format`                 | dtype / regex format validation                          |
| `constant_run`           | **(custom)** ghost-entry detector (>=N identical days)   |
| `distributor_blackout`   | **(custom)** day-wide zero-volume blackout per distributor |
| `credit_cap_signature`   | **(custom, tag-only)** flag credit-limit fingerprints     |

## Modeling — triangulation

We do **not** regress on the censored target. Instead we ensemble three
independently-failing estimators:

1. **Peer ceiling** — KMeans peer-cluster on POI + geo + scale; outlet
   potential = seasonality-adjusted high-quantile of cluster's
   *unconstrained* months × within-cluster size ratio.
2. **Quantile regression** — τ=0.90 quantile regression on outlet
   features (sklearn `QuantileRegressor`, with GBR fallback).
3. **Unconstrained extrapolation** — for outlets with ≥2 unconstrained
   months, use the deflated max of those months × the January 2026
   distributor seasonality index.

The final number is a per-row weight-renormalized blend with a sanity floor
at historical max. See `src/config.py:MODEL_CONFIG` to tune.

## Repository structure

```
.
├── README.md
├── GENAI_LOG.md             # mandatory GenAI transparency log
├── requirements.txt
├── run_pipeline.py
├── data/                    # (untracked) all data lives here
├── notebooks/               # EDA + analytical pipelines
└── src/
    ├── config.py            # single source of truth for paths/schemas/params
    ├── utils/io.py
    ├── ingestion/
    ├── processing/
    ├── features/
    └── modeling/
```

## Notes on raw-data assumptions

The DQ engine, Bronze, and Gold layers all rely on **canonical column names**
defined in `src/utils/io.py:CANONICAL_ALIASES`. The alias map covers many
plausible legacy SFA/ERP column names; if your CSVs use a name not in the
map, either add an alias there or rename the column in `src/config.py`.

## Team

Team **Galaxy Pegasus** — Data Storm v7.0 Preliminary Round.
