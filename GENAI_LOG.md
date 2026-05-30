# GenAI Transparency Log — Team Galaxy Pegasus

This log captures **how, where, and why** Generative AI (LLMs) were used during
the 36-hour Data Storm v7.0 hackathon. It complies with the GenAI Transparency
deliverable in the brief and is updated in real time as work progresses.

> **Rule we follow:** every AI-generated artifact (code, math, prose) is
> reviewed, edited, and validated by a human before it lands in the repo. If
> we rejected an AI suggestion, we say why — that is as informative as what we
> accepted.

---

## Models used

| Model        | Where                  | Purpose                                             |
|--------------|------------------------|-----------------------------------------------------|
| Claude Opus  | Cursor IDE             | Pair-programming, architectural review, refactors   |
| ChatGPT / GPT-5 | Browser              | Brainstorming, math derivation sanity checks         |
| Copilot      | IDE inline             | Boilerplate completion                              |

## Session log

### Phase A — Planning & repo bootstrap
- **Used Claude (Cursor)** to translate the PDF brief into a phased project
  plan and propose the lakehouse folder layout. We kept the layout but
  rejected the suggested over-eager use of Spark/Delta — overkill for 20k
  outlets; pandas + parquet is sufficient and cheaper.
- **Used Claude (Cursor)** to scaffold `src/config.py`, `src/utils/io.py`, and
  the Bronze ingestion module. We **manually verified** schema column names
  against the actual CSVs and corrected the canonical-alias map twice.

### Phase B — Data forensics
- Used **Claude (Cursor)** to run a structured inspection of the 5 raw CSVs,
  cataloguing every schema delta versus our initial assumption. AI proposed
  the dirty-data inventory; we verified each finding interactively in a
  Python shell before adding a corresponding DQ check.
- Caught and documented:
  - **200 outlets with lat/lon swapped** (auto-fixed by `coord_swap_fix`).
  - **40 outlets with (0,0) default coordinates** (rejected via bbox check).
  - **32,240 duplicate transaction rows** on the natural composite key.
  - **4,753 negative volumes** (~0.2%) — kept as tagged returns/credits.
  - **Outlet_Type typos** (`Grocry`, `Bakry`, ` Eatery `, `SMMT`) normalized.
  - **Outlet_Size casing** (`small` vs `Small`) normalized.
  - **Categorical `Seasonality_Index`** (Favorable/Moderate/Un-Favorable)
    converted to numeric multiplier {1.15, 1.00, 0.85}.
  - **Distributor_ID, Province absent from outlet_master** — derived from
    transactions (primary distributor per outlet) and ID-prefix rules.
  - **Holidays PK** corrected from `(date, name)` to `(date, name, type)`
    after discovering legitimate multi-type rows (e.g. Vesak Poya Day is
    Public + Poya Day + Bank + Mercantile).

### Phase C — DQ framework
- AI proposed the registry-based check engine; we kept it but tightened
  the contract: every check returns `(passing, rejected_with_reason)` and
  rejected rows MUST land in `_rejected/` with a non-empty `failure_reason`.
- **AI suggestion rejected:** an initial `low_volume_month_tag` check was
  applied at *transaction-line* grain, producing 63% false-positive
  constraint rate (each transaction line is a single SKU within a month,
  so per-line quantiles are meaningless). We removed it from the DQ pipeline
  and reimplemented monthly-grain constraint flagging inside Gold.
- **AI suggestion rejected:** an early `soft_ceiling_flag` flagged months
  within 20% of the outlet's own max as "constrained" — at monthly grain
  this is the OPPOSITE of what we want (those are exactly the unconstrained
  months that reveal latent demand). Removed.

### Phase D — POI scraping
- **Used Claude (Cursor)** to draft Overpass QL query templates for each POI
  category in `POI_CONFIG.poi_taxonomy`. We hand-tested 3 queries on the
  Overpass turbo web UI before trusting them, and tightened tag filters
  (`amenity~"school|college|kindergarten"` rather than the broader
  `amenity=education` originally suggested, which had patchy OSM coverage in
  rural Sri Lanka).
- Retry/backoff implementation was reviewed for correctness — Overpass
  returns 429 *and* 504 on overload, both need exponential backoff.

### Phase E — Feature engineering
- _to be filled as work progresses._

### Phase F — Latent potential modeling
- **Used ChatGPT** to sanity-check the Tobit MLE derivation and to confirm
  the connection between high-quantile regression and right-side conditional
  envelope estimation. We chose to **prefer quantile regression** in the
  ensemble — Tobit is included as a robustness alternative, not the
  headline method, because Tobit's distributional assumption (Gaussian
  errors below the censoring threshold) is not credible for FMCG volume.
- **Used Claude (Cursor)** to draft the ensemble blending and sensitivity
  table boilerplate. Weights themselves were chosen by the team based on
  the relative coverage and stability of each estimator.

### Phase G — Reporting
- _to be filled as work progresses._

### Phase H — Final round: spatial decay, saturation, transparent model
- **Used Claude (Claude Code)** to implement the Round-2 additions on top of the
  existing lakehouse. The human-authored instruction set the architecture and the
  guardrails; the AI wrote the code to that spec. Key prompt (paraphrased):
  > "Make the transparent multiplicative model the PRIMARY output, built by
  > re-expressing components we already trust. peer_ceiling stays the base;
  > derive constraint_uplift (≥1.0) from the gap between constrained observed
  > volume and unconstrained-deflated stats; pull Jan seasonality OUT into a
  > standalone multiplier; spatial_multiplier = decay-accessibility + saturation,
  > centered at 1.0 and clamped [0.7, 1.4]. Final = peer_ceiling *
  > constraint_uplift * seasonality_index * spatial_multiplier, floored at
  > historical max, capped at peer-cluster max. KEEP the 3-estimator ensemble but
  > DEMOTE it to a validation diagnostic; output per-outlet % divergence as a
  > robustness cross-check and bug detector. Store every factor per outlet for XAI."
- **What the AI built, and the human decisions baked in:**
  - **① Spatial distance-decay (Huff/gravity).** `poi_scraper._features_from_response`
    now emits `poi_access_<cat>_<r>m` = Σ decay-weight, alongside the Round-1
    counts (counts preserved — no regression). Decay scales are **different per
    POI type** (`POI_DECAY_CONFIG.decay_sigma_m`): hospital σ=800 m and tourism
    σ=1000 m pull far; a bus halt σ=150 m is local. Gaussian kernel by default,
    exponential available. These are the gravity model's per-destination β.
  - **② Competitive saturation.** New `src/features/spatial.py`. Own-network
    density via a BallTree haversine query over the coordinate file + OSM shop
    counts (OSM `shop` is competition, so its `demand_weight` is 0 in
    accessibility and it feeds saturation instead). Saturation **adjusts the
    prediction**: it is one of the two inputs to `spatial_multiplier`, which
    discounts saturated clusters and lifts isolated/high-accessibility outlets.
  - **④ Transparent multiplicative model.** `latent_potential_model.estimate_multiplicative`
    multiplies the four stored factors, floors at `hist_total_max`, caps at
    peer-cluster max × uplift cap. The 3-estimator ensemble is retained and run
    every time as `ensemble_potential`; `compute_divergence` reports per-outlet
    `divergence_pct`. Factors are written to `data/gold/outlet_factors.parquet`
    as the XAI input — the narrative layer reads them, never recomputes.
- **Guardrails the human required, AI implemented:** hard clamp on
  `spatial_multiplier` ([0.7, 1.4]); explicit NaN/negative assertion on the final
  prediction (raises, never ships silently); peer-cluster-max cap so a product of
  factors cannot blow up; divergence distribution logged and flagged above 50%.
- **No new magic numbers:** all decay scales, radii, betas, clamp bounds, caps,
  and the budget response `k` live in `src/config.py`.
- **AI suggestion rejected:** computing accessibility per-radius and feeding both
  500 m and 1000 m into the model — redundant, since the decay kernel already
  handles falloff within the widest radius. We emit both columns for audit but the
  model reads only the widest-radius accessibility.

---

## What we explicitly did NOT accept from AI

1. **Imputing missing volume with the mean.** Suggested early; rejected
   because it would have washed out the censoring signal we need to model.
2. **Dropping rejected rows silently.** The brief mandates a quarantine
   store; the AI's first DQ stub used `df.dropna()`. We replaced it.
3. **Using a single XGBoost regressor on historical volume.** This treats
   the censored target as ground truth and would systematically
   under-predict potential for constrained outlets. Rejected in favor of
   the three-method triangulation framework.
4. **Auto-generated commentary in code.** We stripped noise comments like
   `# read the dataframe` — they add no signal and clutter the diff.

---

## Validation discipline

- Every DQ check has a unit-test-style sanity check in `notebooks/`.
- Every modeling assumption (censoring fingerprint, peer-cluster sanity,
  seasonality adjustment) is plotted before being trusted downstream.
- All AI-generated SQL/Overpass queries are run once interactively before
  being added to the pipeline.
