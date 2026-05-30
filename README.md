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

## Modeling — transparent multiplicative model (PRIMARY)

We do **not** regress on the censored target, and we deliberately avoid a black
box so each factor's contribution can be read directly (XAI without
SHAP-on-a-blackbox). The headline prediction is a product of four interpretable,
stored factors:

```
potential = peer_ceiling × constraint_uplift × seasonality_index × spatial_multiplier
```

| Factor | Meaning | Where computed |
|---|---|---|
| `peer_ceiling` | Base: KMeans peer-cluster high-quantile of unconstrained months × within-cluster size ratio | `latent_potential_model.estimate_peer_ceiling` |
| `constraint_uplift` | ≥ 1.0; how left-censored the outlet is (gap between unconstrained-deflated and typical observed, × constrained fraction) | `gold_enrichment.compute_constraint_uplift` |
| `seasonality_index` | Standalone January distributor seasonality multiplier | `gold_enrichment.compute_seasonality_jan_index` |
| `spatial_multiplier` | Bounded [0.7, 1.4]: Huff gravity accessibility (lift) − competitive saturation (discount), centered at 1.0 | `spatial.compute_spatial_multiplier` |

Floored at historical max, capped at peer-cluster max (anti-blow-up). Every
factor is written per outlet to `data/gold/outlet_factors.parquet` — the XAI
layer reads these directly. See `src/config.py` (`MODEL_CONFIG`,
`POI_DECAY_CONFIG`, `SATURATION_CONFIG`, `SPATIAL_CONFIG`) to tune.

### Spatial signals (Huff / gravity)

- **Distance-decay accessibility** (`poi_scraper`): each POI contributes a
  decayed weight `exp(-d²/2σ²)` with a **per-type σ** (hospital/tourism pull
  far; bus halt is local). Replaces flat counts (counts retained for audit).
- **Competitive saturation** (`spatial.py`): our own outlets within radius
  (BallTree haversine) + OSM competitor shops → saturation index that
  discounts crowded catchments and lifts isolated outlets.

### Ensemble — demoted to a validation diagnostic

The Round-1 three-estimator ensemble (peer ceiling + τ=0.90 quantile regression
+ unconstrained extrapolation) is still computed every run as
`ensemble_potential`, as an **independent** cross-check on the transparent model.

**Read the cross-check on rank, not level.** The headline robustness signal is
the **Spearman rank correlation ρ = 0.89** between `mult_potential` and
`ensemble_potential`: two methods built on different principles order the 20k
outlets' potential near-identically. That is the agreement that matters for a
relative-ranking deliverable.

The two models differ in **level** by design — the median absolute
`divergence_pct` is ~93%, and this is **structural, not error**:

- The **ensemble is a conservative, censored band.** Two of its three
  estimators (quantile regression, unconstrained extrapolation) are pulled
  toward historically *observed* volume, which is left-censored — so the
  ensemble partly inherits the very ceiling-suppression we are trying to undo.
- The **transparent model is an uncapped ceiling.** It anchors on the peer
  ceiling and lifts it by the constraint / seasonality / spatial factors.

So `mult_potential ≈ 2× ensemble_potential` is the expected gap between an
*uncapped demand ceiling* and a *censored conservative band*, not a defect. We
deliberately do **not** report a median-rescaled divergence — forcing the two to
a common median would be circular (it assumes the very level agreement we are
testing). `divergence_flag` (|divergence| > `divergence_flag_pct`) is retained
purely as a **per-outlet defect detector**: it surfaces individual outlets where
the transparent factor product disagrees with the ensemble far more than the
~2× structural offset, which usually means a bad input for that outlet.

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
