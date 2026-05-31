# Unmasking Latent Outlet Potential under Left-Censored Demand
### Data Storm v7.0 — Final Round · Technical Methodology Paper

**Team Galaxy Pegasus**

> Conversion note: this Markdown is the source for the ≤10-page PDF. Suggested
> mapping — Cover (p1), §1 (p1–2), §2 (p2–4), §3 (p4–5), §4 (p5–7), §5 (p7–8),
> §6 (p8–9), §7 (p9–10). Export with Pandoc:
> `pandoc docs/TECHNICAL_PAPER.md -o paper.pdf --toc -V geometry:margin=2cm`

---

## Cover

- **Challenge:** Estimate the *Maximum Monthly Purchase Potential* (liters,
  January 2026) for **20,000** traditional-trade outlets across **4 provinces**
  and **10 distributors** in Sri Lanka, then allocate a **LKR 5,000,000** trade
  budget across the Western Province to maximize incremental volume.
- **Core difficulty:** Historical sales are **left-censored** —
  `observed = min(true_demand, systemic_constraint)` (credit caps, stockouts,
  route limits). There is **no ground-truth target**; evaluation is qualitative,
  on defensibility.
- **Our answer:** a deliberately **transparent, multiplicative** potential model
  whose every factor is inspectable, cross-checked against an independent
  ensemble, and surfaced in a web app and a concave-optimal budget allocator.

---

## 1. Approach at a glance

We reject regressing on the censored target — that would teach a model to
reproduce the very ceilings we must remove. Instead we estimate a **demand
ceiling** as a transparent product of four interpretable factors:

```
potential = peer_ceiling × constraint_uplift × seasonality_index × spatial_multiplier
            └ base ┘       └ uncapping ┘        └ timing ┘          └ location ┘
```

floored at the outlet's historical max and capped at its peer-cluster max. The
design is intentionally **not** a black box: because the model is a product of
named factors, the "explanation" of any prediction is the factors themselves —
no SHAP, no surrogate model. An **independent 3-estimator ensemble** is computed
in parallel as a robustness cross-check.

**Headline results (20,000 outlets):** median potential **972 L/mo**, total
**20.4M L/mo**; **96.1%** of outlets lifted above their own historical ceiling;
transparent vs. ensemble **Spearman ρ = 0.89**.

**Pipeline:** a Bronze→Silver→Gold lakehouse (pandas + parquet), idempotent and
re-runnable end-to-end via `python run_pipeline.py`.

---

## 2. Data Engineering & Scraping Pipeline (brief §5a)

### 2.1 Lakehouse architecture

| Layer | Module | Responsibility |
|---|---|---|
| Bronze | `ingestion/bronze_ingestion.py` | Raw CSV → parquet, **SHA-256 manifest** for auditability, column canonicalization only |
| DQ | `processing/data_quality.py` | Reusable, registry-based check engine; every check returns `(passing, rejected_with_reason)` |
| Silver | `processing/silver_cleaning.py` | Apply per-dataset check pipelines, **quarantine** rejects, enrichment joins |
| POI | `features/poi_scraper.py` | Overpass API scrape + cache + distance-decay features |
| Spatial | `features/spatial.py` | Gravity accessibility + competitive saturation |
| Gold | `features/gold_enrichment.py` | Outlet×month aggregation + outlet-level factor table |
| Model | `modeling/latent_potential_model.py` | Transparent model + ensemble diagnostic |
| Optimize | `optimization/budget_allocation.py` | Western-Province trade-spend allocation |

Inputs (2023–2025): **2,376,389** transaction rows, 20,000 outlets, 20,000
coordinate rows, distributor seasonality, holiday calendar.

### 2.2 External POI acquisition (OpenStreetMap / Overpass)

We scraped OSM POIs around every outlet via the **Overpass API**, with an
engineering posture built for a flaky public endpoint:

- **13-category taxonomy** (school, university, bus stand/station, railway,
  hospital, place of worship, tourism, market, government, restaurant, sports,
  shop), each a tuned Overpass tag filter.
- **Two radii** (500 m, 1000 m). **On-disk JSON cache** keyed by
  `outlet × radius` — **37,423 cached responses** — so the pipeline resumes
  after rate limits and re-runs offline.
- **Exponential backoff** on 429/502/503/504; polite User-Agent; deterministic
  cache keys.

### 2.3 Features engineered to proxy footfall & market potential

We do **not** use flat POI counts as the demand signal. Instead:

- **Distance-decay accessibility (Huff / gravity).** Each POI contributes a
  decayed weight `w(d) = exp(−d² / 2σ²)`, summed per type. Crucially the decay
  scale **σ differs per POI type** — a hospital (σ=800 m) or tourist attraction
  (σ=1000 m) pulls demand from far away, while a bus halt (σ=150 m) is hyper-local.
  Each type also carries a Huff **attractiveness weight** `A_j` (a market/transit
  node drives more impulse traffic than a courthouse). Raw counts are retained
  for audit but are not the modeling signal.
- **Competitive saturation.** How many *other sellers* share a catchment:
  our own neighbouring outlets (counted with a **BallTree haversine** query over
  the coordinate file) plus third-party OSM shops. High saturation ⇒ demand is
  split across more sellers.

These combine into the bounded `spatial_multiplier` (§4.4). All decay scales,
radii, and weights live in `config.py` — no magic numbers.

---

## 3. Data Cleaning (brief §5b)

### 3.1 Initial quality assessment

Forensic EDA surfaced the classic legacy-SFA artifacts: lat/long transpositions,
placeholder `(0,0)` coordinates, duplicate invoice rows on the natural key,
negative volumes (returns/credits), `Outlet_Type` typos (`Grocry`, `Bakry`,
`SMMT`), `Outlet_Size` casing variants, a categorical `Seasonality_Index`, and
`Distributor_ID`/`Province` absent from the outlet master.

### 3.2 Programmatic cleaning & artifact neutralization

A **declarative, reusable DQ engine** applies a per-dataset pipeline of checks;
every rejected row is **quarantined** to `data/silver/_rejected/` with a
`failure_reason` — **never silently dropped**. Results on the production run:

| Dataset | Action | Outcome |
|---|---|---|
| Coordinates | Auto-fix lat/long swaps in place; bbox-validate against Sri Lanka | **40** out-of-bounds coords quarantined; 19,960 geocoded |
| Transactions | Dedupe on `(outlet, year, month, distributor, sku)` | **32,240** duplicates quarantined → 2,344,149 clean |
| Transactions | **Tag** negative volumes as returns (not reject) | preserved as signal |
| Outlets | Normalize `Outlet_Type`/`Outlet_Size` typos & casing | 20,000 canonicalized |
| Holidays | PK = `(date, name, type)` (multi-type days are legitimate) | **93** true duplicates quarantined |
| Seasonality | Map categorical → numeric `{Favorable 1.15, Moderate 1.00, Un-Favorable 0.85}` | 360 rows |

**Enrichment:** `distributor_id` derived per outlet from transactions, `province`
from the distributor-ID prefix, coordinates joined to the master.

### 3.3 Neutralizing the censoring artifacts (not just dirt)

Beyond cleaning, we **fingerprint systemic constraints** at the outlet-month
grain so the model can treat constrained months differently: a stockout proxy
(month far below the outlet's own median), single-SKU months (route restriction),
and a **credit-cap signature** (volumes that are suspiciously clean multiples of
50/100/500/1000). These are *tagged*, not dropped — they are the evidence of
censoring we later exploit.

---

## 4. The Mathematical Framework (brief §5c)

### 4.1 The censoring problem

Observed volume is `y_obs = min(y_true, c)` where `c` is an unobserved systemic
constraint. The conditional mean `E[y_obs | x]` is biased **downward** wherever
`c` binds, so any mean-regression under-predicts exactly the constrained outlets
we care about. We therefore estimate an **upper envelope / ceiling**, not a mean.

### 4.2 Factor 1 — `peer_ceiling` (the base)

Outlets are clustered (KMeans) on spatial + scale features. An outlet's base
ceiling is a **high quantile (p90) of its peer cluster's unconstrained,
seasonality-deflated months**, rescaled by the outlet's own size within the
cluster. Rationale: the best *unconstrained* months of *comparable* outlets
reveal the demand a censored outlet could reach. *Distribution:* p10 353,
p50 816, p90 1453, max 1761 L.

### 4.3 Factor 2 — `constraint_uplift ≥ 1.0` (the uncapping)

Derived purely from already-computed signals:

```
constrained_fraction = months_constrained / months_observed
gap_ratio            = clip(unc_deflated_mean / hist_total_mean, 1, GAP_CAP)
constraint_uplift    = clip(1 + constrained_fraction · (gap_ratio − 1), 1, UPLIFT_CAP)
```

The lift scales with **how often** an outlet is constrained **and** the gap
between its unconstrained behaviour and its typical observed level. An
unconstrained outlet gets exactly 1.0 (no change). *Distribution:* **90.6%** of
outlets receive a lift; p50 1.03, p90 1.11, cap 1.60.

### 4.4 Factor 3 — `seasonality_index` (timing)

The January distributor seasonality index, **pulled out as a standalone
multiplier** (it used to be buried inside estimators) so the XAI layer can
attribute it cleanly. January 2026 is absent from the data, so we use the
January-2025 index as a proxy. *Distribution:* 17,000 outlets at 1.00, 3,000 at
1.15 (favorable-January distributors).

### 4.5 Factor 4 — `spatial_multiplier` (location), bounded by design

```
raw = access_beta · z(log accessibility) − sat_beta · z(log saturation)
spatial_multiplier = clip(1 + raw − median(raw), 0.70, 1.40)
```

Robust (median/IQR) standardization centers it at **1.0**: accessibility lifts,
saturation discounts. The hard clamp `[0.70, 1.40]` is a deliberate guardrail —
in a *product* of factors an unbounded term can explode the result.
*Distribution:* p10 0.78, p50 1.00, p90 1.36; only **2.65%** pinned at the floor
and **8.3%** at the ceiling (89% strictly interior — no clamp pile-up).

### 4.6 Assembly, guardrails, and why it's defensible

`potential = ∏ factors`, then **floored at historical max** (potential can't be
below demonstrated sales) and **capped at peer-cluster max** (anti-blow-up).
Guardrails assert **no NaN, no negatives**. Result: **96.1%** of outlets are
lifted above their own historical ceiling (the model genuinely *uncaps*), while
**3.9% (778)** fall back to the historical-max floor. The typical outlet sits
**+13% above its peer_ceiling** (median `mult/peer_ceiling = 1.13`).

### 4.7 Robustness cross-check — transparent vs. independent ensemble

We also run the Round-1 **3-estimator ensemble** (peer ceiling + τ=0.90 quantile
regression + unconstrained extrapolation) and compare:

- **Rank agreement is the headline: Spearman ρ = 0.89** across all 20,000
  outlets — two methods built on different principles order the outlets'
  potential near-identically.
- **Level differs by design** (median |divergence| ≈ 93%, i.e. `mult ≈ 2×
  ensemble`). This is **structural, not error**: the ensemble is a *conservative,
  censored band* (two of its estimators are pulled toward censored observed
  volume), while the transparent model is an *uncapped ceiling*. We deliberately
  do **not** median-rescale the comparison (that would be circular). A per-outlet
  `divergence_flag` repurposes the metric as a **defect detector** for individual
  outlets that disagree far beyond the structural offset.

---

## 5. Spend Optimization Logic (brief §5d)

**Objective:** distribute LKR 5,000,000 across Western-Province outlets to
maximize incremental January-2026 volume vs. normal sales.

- **Headroom:** `headroom_i = max(potential_i − historical_i, 0)`
  (`historical_i` = `hist_total_median`). Western totals: **9,000 outlets**,
  **6.75M L** of headroom.
- **Concave response:** `lift_i(s) = headroom_i · (1 − e^{−k·s})`. The first
  rupees buy the most lift; returns diminish; lift asymptotes at headroom (you
  cannot sell past true demand).
- **Greedy marginal allocation:** repeatedly fund the outlet with the highest
  marginal lift `headroom_i · e^{−k·s_i} · (1 − e^{−k·step})`. Because the
  objective is **concave and separable**, greedy is **provably globally optimal**
  — and fully explainable (every rupee has a one-line marginal-value reason).
- **Discrete real-world constraints:** spend is **lumpy** (multiples of
  `step_lkr`); **coolers are integer** (`cooler_cost` = a whole number of steps);
  a **per-outlet cap** prevents over-investing past diminishing returns.
- **Guardrails:** total ≤ budget, non-negative, multiple-of-step — all asserted.

**Result (at k = 5×10⁻⁶):** **238** outlets funded, exactly **LKR 5,000,000**
allocated, projected **+44,940 L** incremental (≈111 LKR/L), balanced across the
three Western distributors (DIST_W_01/02/03: 73/91/74 outlets).

> **Honest caveat:** `k` is a **business assumption** (promo elasticity), not
> fitted from data. It scales the absolute volume and the concentration of spend;
> the *relative ranking* of where to spend is robust to it. Calibrate `k` to the
> client's observed promo response before quoting a headline volume.

---

## 6. GenAI Transparency Log (brief §5e)

Full detail in **`GENAI_LOG.md`** (live, phase-by-phase). Summary:

- **How/where:** Claude (Claude Code / Cursor) for scaffolding, refactors, and
  implementing human-specified designs; ChatGPT for math sanity-checks (Tobit /
  quantile-envelope reasoning); Copilot for boilerplate.
- **Discipline:** every AI artifact is human-reviewed before landing; rejected
  suggestions are logged with the reason (it's as informative as what we kept).
- **Most effective prompts** are quoted verbatim in the log — e.g. the
  transparent-model spec (Phase H), the greedy-optimality justification we asked
  AI to *prove then we verified* (Phase I), and the app's factor-waterfall design
  (Phase J).
- **Explicitly rejected:** mean-imputation of missing volume (washes out the
  censoring signal), silently dropping rejects (we quarantine), and a single
  black-box XGBoost on censored volume (would entrench the ceilings).

---

## 7. Outlet Intelligence App, Limitations & Reproducibility

- **App** (`app/streamlit_app.py`): browse 20k predictions; filter by province /
  distributor / type / ID; **drill into any outlet** to see its
  factor-decomposition waterfall (the transparent model's payoff), spatial
  drivers, ensemble cross-check, and recommended spend. Run: `streamlit run
  app/streamlit_app.py`.
- **Limitations:** (i) `k` is assumed, not fitted; (ii) OSM POI coverage is
  uneven in rural areas; (iii) Jan-2026 seasonality proxied by Jan-2025;
  (iv) constraint flags are heuristics, not labels.
- **Reproducibility:** single command `python run_pipeline.py` runs Bronze →
  Silver → POI → Gold → Model → Budget; POI is cached; all parameters in
  `src/config.py`; DQ rejects auditable in `data/silver/_rejected/`.

*All figures in this paper were produced by the committed pipeline on the full
20,000-outlet dataset.*
