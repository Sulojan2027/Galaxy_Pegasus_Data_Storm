# Executive Pitch Deck — Outline
### Data Storm v7.0 Final · Team Galaxy Pegasus · **≤10 slides**

> C-suite audience, **no mathematical jargon**. 10-minute pitch + 5-minute live
> demo. Each slide below = one deck slide: **headline**, **talking points**,
> **suggested visual**. Numbers are validated from the production pipeline run.

---

### Slide 1 — Title & the one-sentence promise
- **Headline:** *"Seeing the sales we've been missing — and spending smarter to win them."*
- **Talking points:**
  - We estimated the true monthly potential of all **20,000** outlets for Jan 2026.
  - And turned it into a **LKR 5M trade-spend plan** that maximizes extra volume.
- **Visual:** Team name + a Sri Lanka map dotted with 20k outlets.

---

### Slide 2 — The hidden problem: our history lies to us
- **Headline:** *"Past sales show what outlets BOUGHT — not what they could SELL."*
- **Talking points:**
  - When an outlet runs out of stock, hits a credit limit, or gets skipped on a
    route, the books record a *low* number — not real demand.
  - Planning from that history means we keep under-serving our best outlets.
  - We needed to **un-cap** the data to see the real opportunity.
- **Visual:** simple line — "true demand" flat-topped by a "ceiling," with the
  shaded gap labelled *"the volume we never recorded."*

---

### Slide 3 — Our approach, in plain English
- **Headline:** *"We build each outlet's potential from four things you can see."*
- **Talking points (no formulas):**
  1. **Peers** — what similar outlets achieve on their best, unconstrained months.
  2. **Suppression** — how often this outlet was held back, and by how much.
  3. **Timing** — the January seasonal effect.
  4. **Location** — nearby footfall (schools, transit, markets) vs. competition.
  - Deliberately a **glass box, not a black box**: every number is explainable.
- **Visual:** 4 labelled icons → multiply into one "Potential" figure.

---

### Slide 4 — It works, and we can prove it
- **Headline:** *"Two independent methods agree on the ranking — 0.89 correlation."*
- **Talking points:**
  - A second, fully separate model ranks outlets almost identically (**ρ = 0.89**).
  - **96%** of outlets show real upside above their own historical best — we are
    genuinely uncovering demand, not inflating numbers.
  - Built-in guardrails stop any outlet's estimate from running away.
- **Visual:** scatter of Method A vs Method B (tight diagonal) + "96% uncapped" stat.

---

### Slide 5 — Every prediction explains itself (live-demo hook)
- **Headline:** *"Click any outlet — see exactly why."*
- **Talking points:**
  - Business users get a web app: browse, filter by province/distributor, and
    **drill into one outlet** to see the four drivers stacked up.
  - No data-science team needed to interpret a score.
- **Visual:** screenshot of the app's **factor waterfall** for one outlet.

---

### Slide 6 — From insight to money: the LKR 5M question
- **Headline:** *"Where do 5 million rupees buy the most extra volume?"*
- **Talking points:**
  - Western Province: **9,000 outlets**, **6.75M L** of untapped headroom.
  - Spending has **diminishing returns** — the 10th poster does less than the 1st.
  - So we fund the highest-return outlet *rupee by rupee* until the money's gone.
- **Visual:** a diminishing-returns curve (spend → extra volume, flattening).

---

### Slide 7 — The allocation, and why it's optimal
- **Headline:** *"A focused, fair, mathematically-optimal plan."*
- **Talking points:**
  - Plan funds **238** highest-opportunity outlets, **balanced across all three
    Western distributors** (73 / 91 / 74).
  - Method is **provably optimal** for this kind of problem — and every rupee's
    placement has a one-line justification.
  - Respects reality: whole coolers, lumpy merchandising packs, per-outlet caps.
- **Visual:** bar — spend & funded-outlet count per distributor.

---

### Slide 8 — The business impact
- **Headline:** *"+~45,000 L of incremental monthly volume from the same budget."*
- **Talking points:**
  - Projected **+44,940 L/month** incremental from the LKR 5M (≈111 LKR per
    extra liter) at our baseline assumption.
  - Concentrated where it converts — not peanut-buttered across 9,000 outlets.
  - **Caveat we'll own:** the volume figure scales with promo-response strength;
    we'll calibrate it to your real campaign data before locking targets.
- **Visual:** before/after volume bar + "5M LKR → +45k L" arrow.

---

### Slide 9 — Rolling it out on the ground
- **Headline:** *"A route-ready list, not a research paper."*
- **Talking points:**
  - Output is a **per-outlet spend list** per distributor — hand it straight to
    sales reps.
  - Reps open the app, see each funded outlet's *why*, and act.
  - Re-runs monthly as new data lands; budget and assumptions are dials, not code.
- **Visual:** flow — *Model → ranked outlet list → distributor → rep → outlet.*

---

### Slide 10 — Close & ask
- **Headline:** *"Glass-box potential. Optimal spend. Ready to deploy."*
- **Talking points:**
  - Defensible (two methods agree), explainable (every score decomposes),
    actionable (a spend list today).
  - **Ask:** pilot the Western plan for January; calibrate `k` on results; scale
    to all four provinces.
- **Visual:** the 3 pillars (Defensible / Explainable / Actionable) + next-step timeline.

---

## Live-demo script (5 min)
1. Open app → **Browse**: filter to Western, sort by potential (10s).
2. **Drill-down** on a top outlet → walk the factor waterfall in plain words (90s).
3. Show the **ensemble cross-check** line on the same outlet (30s).
4. Switch to **Budget allocation** tab → show funded outlets + distributor split (60s).
5. Filter to a single distributor → "this is what that rep gets Monday" (30s).
6. Buffer for Q&A handoff.

## Numbers cheat-sheet (all validated from the pipeline)
- 20,000 outlets · median potential **972 L/mo** · total **20.4M L/mo**
- **96.1%** uncapped above historical max · typical **+13%** above peer baseline
- Cross-check **Spearman ρ = 0.89**
- Western: **9,000** outlets · **6.75M L** headroom · **238** funded · **+44,940 L** · **111 LKR/L**
- Budget: exactly **LKR 5,000,000**, balanced 73/91/74 across DIST_W_01/02/03
