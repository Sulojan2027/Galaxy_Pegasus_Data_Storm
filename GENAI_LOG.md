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
- _to be filled as work progresses._

### Phase C — DQ framework
- _to be filled as work progresses._

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
