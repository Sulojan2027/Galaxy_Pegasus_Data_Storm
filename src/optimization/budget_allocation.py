"""Marketing Spend Optimization — Western Province trade-spend allocation.

Problem (brief §2.3): a fixed **LKR 5,000,000** promotional budget for the
Western Province (Jan 2026) must be distributed across outlets to **maximize
incremental sales volume vs. normal historical sales**, without exceeding the
budget.

Model
-----
For each outlet *i* we define **headroom** — the volume it could gain if fully
supported — as the gap between its uncapped potential and its normal monthly
sales:

    headroom_i = max(potential_i - historical_i, 0)

where ``potential_i`` is the transparent model's ``mult_potential`` (Jan 2026)
and ``historical_i`` is the outlet's normal monthly volume (median by default).

Spending *s* LKR on an outlet yields incremental volume through a **concave,
diminishing-returns response function**:

    lift_i(s) = headroom_i * (1 - exp(-k * s))

This is the right shape: the first rupees on an outlet buy a lot of lift; each
additional rupee buys less; lift asymptotes at the outlet's headroom (you cannot
sell past true demand). ``k`` (curvature) is in ``BUDGET_CONFIG``.

Optimization — greedy marginal allocation
------------------------------------------
The objective Σ lift_i(s_i) is **concave and separable**, and the constraint set
(Σ s_i ≤ B, 0 ≤ s_i ≤ cap, s_i a multiple of the lumpy step) is a matroid-like
polytope. For such problems **greedy marginal allocation is globally optimal**:
repeatedly give the next lumpy increment of spend to whichever outlet currently
has the highest *marginal* lift

    dLift_i = headroom_i * exp(-k * s_i) * (1 - exp(-k * step))

Because ``(1 - exp(-k*step))`` is constant across outlets, this is equivalent to
always funding the outlet with the largest ``headroom_i * exp(-k * s_i)`` — i.e.
high-headroom outlets first, then spreading out as their marginal returns decay
below those of the next outlet. A max-heap makes this O(#steps · log n).

Greedy is not just convenient here — it is **provably optimal** for a concave
separable objective, and it is fully **explainable** (every rupee's placement has
a one-line marginal-value justification), which the qualitative evaluation rewards.

Discrete real-world constraints
-------------------------------
- **Merchandising is lumpy:** spend moves in ``step_lkr`` increments (e.g. you
  buy a poster/standee pack, not LKR 137 of one). All allocations are therefore
  integer multiples of ``step_lkr`` by construction.
- **Coolers are integer:** ``cooler_cost_lkr`` is a whole multiple of
  ``step_lkr`` (50,000 / 5,000 = 10 steps), so an integer number of coolers is
  representable exactly as a block of steps — no fractional coolers can occur.
- **Per-outlet cap:** no single outlet may absorb more than
  ``max_spend_per_outlet_lkr`` (diminishing returns + fairness across the route).

Outputs
-------
- ``data/predictions/<team>_budget_allocations.csv`` — the deliverable:
  ``Outlet_ID, Trade_Spend_Allocation_LKR`` for every Western outlet.
- ``data/predictions/budget_allocation_diagnostics.parquet`` — per-outlet
  headroom, spend, expected incremental volume, saturation fraction (for the
  paper / app).
- ``data/predictions/budget_allocation_by_distributor.csv`` — distributor roll-up.
"""

from __future__ import annotations

import heapq
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.utils.io import read_parquet, setup_logging, write_csv, write_parquet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_allocation_inputs(
    gold_dir: Path,
    baseline_col: str = "hist_total_median",
) -> pd.DataFrame:
    """Join potential (factor table) with the historical baseline + geography
    (feature table), filtered to the target province.

    Returns one row per target-province outlet with ``outlet_id``,
    ``distributor_id``, ``potential``, ``historical``, ``headroom``.
    """
    feats = read_parquet(gold_dir / "outlet_features.parquet")
    facs = read_parquet(gold_dir / "outlet_factors.parquet")

    province = config.BUDGET_CONFIG["target_province"]
    keep = ["outlet_id", "distributor_id", "province", baseline_col, "cooler_count"]
    keep = [c for c in keep if c in feats.columns]
    df = feats[keep].copy()
    df = df[df["province"] == province] if "province" in df.columns else df

    pot = facs[["outlet_id", "mult_potential"]].rename(columns={"mult_potential": "potential"})
    df = df.merge(pot, on="outlet_id", how="left")

    df["historical"] = pd.to_numeric(df.get(baseline_col), errors="coerce").fillna(0.0).clip(lower=0.0)
    df["potential"] = pd.to_numeric(df["potential"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["headroom"] = (df["potential"] - df["historical"]).clip(lower=0.0)

    logger.info(
        "allocation inputs: %d %s-province outlets | total headroom %.0f L | "
        "%d with positive headroom",
        len(df), province, df["headroom"].sum(), int((df["headroom"] > 0).sum()),
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Greedy marginal allocation (provably optimal for the concave objective)
# ---------------------------------------------------------------------------
def greedy_allocate(
    headroom: np.ndarray,
    budget: float,
    k: float,
    step: float,
    cap: float,
) -> np.ndarray:
    """Allocate ``budget`` in lumpy ``step`` increments to maximize
    Σ headroom_i (1 - exp(-k s_i)) subject to s_i ≤ cap.

    Returns the per-outlet spend vector (each entry a multiple of ``step``).
    """
    n = len(headroom)
    spend = np.zeros(n, dtype=float)
    if n == 0 or budget < step:
        return spend

    step_factor = 1.0 - math.exp(-k * step)  # constant across outlets
    # Max-heap via negated marginal gain. Initial marginal gain at s=0.
    heap: list[tuple[float, int]] = []
    for i in range(n):
        if headroom[i] > 0:
            mg = headroom[i] * step_factor  # exp(-k*0) = 1
            heap.append((-mg, i))
    heapq.heapify(heap)

    allocated = 0.0
    eps = 1e-9
    while heap and allocated + step <= budget + eps:
        neg_mg, i = heapq.heappop(heap)
        if spend[i] + step > cap + eps:
            continue  # outlet capped — drop it, do not consume budget
        spend[i] += step
        allocated += step
        # Re-push with decayed marginal gain if it can still take another step.
        if spend[i] + step <= cap + eps:
            mg = headroom[i] * math.exp(-k * spend[i]) * step_factor
            heapq.heappush(heap, (-mg, i))
    logger.info("greedy allocated %.0f of %.0f LKR (%.1f%%)", allocated, budget, 100 * allocated / budget)
    return spend


def expected_incremental_volume(headroom: np.ndarray, spend: np.ndarray, k: float) -> np.ndarray:
    """Realized lift per outlet: headroom_i * (1 - exp(-k s_i))."""
    return headroom * (1.0 - np.exp(-k * spend))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_budget_optimization(
    gold_dir: Path | None = None,
    predictions_dir: Path | None = None,
    team_name: str = "galaxy_pegasus",
    baseline_col: str = "hist_total_median",
) -> dict[str, Any]:
    setup_logging()
    gold_dir = Path(gold_dir or config.GOLD_DIR)
    predictions_dir = Path(predictions_dir or config.PREDICTIONS_DIR)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    cfg = config.BUDGET_CONFIG
    budget = float(cfg["total_budget_lkr"])
    k = float(cfg["response_k"])
    step = float(cfg["step_lkr"])
    cap = float(cfg["max_spend_per_outlet_lkr"])

    df = load_allocation_inputs(gold_dir, baseline_col=baseline_col)
    if df.empty:
        raise ValueError("no target-province outlets found — cannot allocate budget")

    spend = greedy_allocate(df["headroom"].to_numpy(), budget, k, step, cap)
    df["trade_spend_lkr"] = spend
    df["expected_incremental_liters"] = expected_incremental_volume(
        df["headroom"].to_numpy(), spend, k
    )
    df["headroom_captured_frac"] = np.where(
        df["headroom"] > 0,
        df["expected_incremental_liters"] / df["headroom"],
        0.0,
    )

    # ---- Guardrails ----
    total_spend = float(df["trade_spend_lkr"].sum())
    if total_spend > budget + 1e-6:
        raise ValueError(f"allocation {total_spend:.2f} exceeds budget {budget:.2f}")
    if (df["trade_spend_lkr"] < -1e-9).any():
        raise ValueError("negative allocation produced")
    # lumpiness: every allocation is a multiple of step
    rema = np.mod(df["trade_spend_lkr"].to_numpy() + 1e-6, step)
    if (np.minimum(rema, step - rema) > 1e-3).any():
        raise ValueError("allocation not a multiple of the lumpy step")

    # ---- Deliverable CSV (Outlet_ID + Trade Spend Allocation LKR) ----
    out = df[["outlet_id", "trade_spend_lkr"]].rename(
        columns={"outlet_id": "Outlet_ID", "trade_spend_lkr": "Trade_Spend_Allocation_LKR"}
    )
    out["Trade_Spend_Allocation_LKR"] = out["Trade_Spend_Allocation_LKR"].round(2)
    csv_path = predictions_dir / f"{team_name}_budget_allocations.csv"
    write_csv(out, csv_path)

    # ---- Per-outlet diagnostics (for paper / app) ----
    diag_path = predictions_dir / "budget_allocation_diagnostics.parquet"
    write_parquet(df, diag_path)

    # ---- Distributor roll-up ----
    by_dist = (
        df.groupby("distributor_id")
        .agg(
            outlets=("outlet_id", "size"),
            outlets_funded=("trade_spend_lkr", lambda s: int((s > 0).sum())),
            spend_lkr=("trade_spend_lkr", "sum"),
            incremental_liters=("expected_incremental_liters", "sum"),
        )
        .reset_index()
    )
    by_dist["lkr_per_incremental_liter"] = (
        by_dist["spend_lkr"] / by_dist["incremental_liters"].replace(0, np.nan)
    ).round(2)
    dist_csv = predictions_dir / "budget_allocation_by_distributor.csv"
    write_csv(by_dist, dist_csv)

    funded = int((df["trade_spend_lkr"] > 0).sum())
    total_incr = float(df["expected_incremental_liters"].sum())
    summary = {
        "allocations_csv": csv_path,
        "diagnostics_parquet": diag_path,
        "by_distributor_csv": dist_csv,
        "province": cfg["target_province"],
        "budget_lkr": budget,
        "allocated_lkr": total_spend,
        "n_outlets_total": int(len(df)),
        "n_outlets_funded": funded,
        "total_incremental_liters": round(total_incr, 1),
        "blended_lkr_per_incremental_liter": round(total_spend / total_incr, 2) if total_incr else None,
        "spend_p50_funded": float(df.loc[df.trade_spend_lkr > 0, "trade_spend_lkr"].median()) if funded else 0.0,
        "max_spend_outlet": float(df["trade_spend_lkr"].max()),
    }
    logger.info(
        "budget optimization: funded %d/%d outlets, %.0f LKR -> +%.0f L (%.2f LKR/L)",
        funded, len(df), total_spend, total_incr,
        summary["blended_lkr_per_incremental_liter"] or 0.0,
    )
    return summary


if __name__ == "__main__":
    run_budget_optimization()
