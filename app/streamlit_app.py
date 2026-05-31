"""Outlet Intelligence — Streamlit web app (Data Storm v7.0 deliverable #4).

Lets a business user:
  (a) browse outlet-level Jan-2026 potential across the full 20k dataset,
  (b) filter by province and/or distributor (and outlet type / id search),
  (c) drill into one outlet and see the *reasoning* behind its score.

The "reasoning" is a direct decomposition of the transparent model
    potential = peer_ceiling × constraint_uplift × seasonality_index × spatial_multiplier
read straight from ``data/gold/outlet_factors.parquet`` — no black box, no SHAP.

Run:
    streamlit run app/streamlit_app.py
(Reads the parquet/CSV outputs of the pipeline; run ``python run_pipeline.py`` first.)
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from llm_explainer import generate_outlet_explanation

# Make ``src`` importable when launched via ``streamlit run`` from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

st.set_page_config(page_title="Outlet Intelligence — Data Storm v7.0", layout="wide")

FACTOR_COLS = ["peer_ceiling", "constraint_uplift", "seasonality_jan_index", "spatial_multiplier"]


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading outlet intelligence…")
def load_master() -> pd.DataFrame:
    """Join gold features + factor table + budget allocation into one frame."""
    gold = config.GOLD_DIR
    preds = config.PREDICTIONS_DIR

    factors = pd.read_parquet(gold / "outlet_factors.parquet")

    feat_cols = [
        "outlet_id", "province", "distributor_id", "outlet_type", "outlet_size",
        "latitude", "longitude", "cooler_count",
        "hist_total_median", "hist_total_mean", "hist_total_max",
        "months_observed", "months_constrained", "months_unconstrained",
        "poi_accessibility", "saturation_index", "own_network_density",
    ]
    feats = pd.read_parquet(gold / "outlet_features.parquet")
    feats = feats[[c for c in feat_cols if c in feats.columns]]

    df = feats.merge(factors, on="outlet_id", how="left")

    # Budget allocation (Western only) — optional.
    alloc_path = preds / "galaxy_pegasus_budget_allocations.csv"
    diag_path = preds / "budget_allocation_diagnostics.parquet"
    if alloc_path.exists():
        alloc = pd.read_csv(alloc_path).rename(columns={"Outlet_ID": "outlet_id"})
        df = df.merge(alloc, on="outlet_id", how="left")
    if diag_path.exists():
        diag = pd.read_parquet(diag_path)[
            ["outlet_id", "headroom", "expected_incremental_liters", "headroom_captured_frac"]
        ]
        df = df.merge(diag, on="outlet_id", how="left")

    # Convenience: raw factor product (pre floor/cap) to detect binding guardrails.
    if set(FACTOR_COLS).issubset(df.columns):
        df["raw_factor_product"] = df[FACTOR_COLS].prod(axis=1)
    return df


def _fmt(x: float, unit: str = "") -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}{unit}"


# ---------------------------------------------------------------------------
# Guard: data present?
# ---------------------------------------------------------------------------
if not (config.GOLD_DIR / "outlet_factors.parquet").exists():
    st.error(
        "No model outputs found. Run the pipeline first:\n\n"
        "```\npython run_pipeline.py\n```"
    )
    st.stop()

df = load_master()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")

# --- AI explainability key (optional) ---
with st.sidebar.expander("🤖 AI Explanations (optional)", expanded=False):
    _env_key = __import__("os").environ.get("GROQ_API_KEY", "")
    try:
        _secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        _secret_key = ""
    _default_key = _env_key or _secret_key
    groq_api_key = st.text_input(
        "Groq API key",
        value=_default_key,
        type="password",
        help="Get a free key at console.groq.com. Used only to generate outlet narratives.",
    )


provinces = sorted(df["province"].dropna().unique().tolist()) if "province" in df else []
sel_prov = st.sidebar.multiselect("Province", provinces, default=provinces)

prov_mask = df["province"].isin(sel_prov) if sel_prov else pd.Series(True, index=df.index)
dist_opts = sorted(df.loc[prov_mask, "distributor_id"].dropna().unique().tolist())
sel_dist = st.sidebar.multiselect("Distributor", dist_opts, default=dist_opts)

type_opts = sorted(df["outlet_type"].dropna().unique().tolist()) if "outlet_type" in df else []
sel_type = st.sidebar.multiselect("Outlet type", type_opts, default=type_opts)

search_id = st.sidebar.text_input("Search Outlet_ID contains").strip()

mask = prov_mask.copy()
if sel_dist:
    mask &= df["distributor_id"].isin(sel_dist)
if sel_type:
    mask &= df["outlet_type"].isin(sel_type)
if search_id:
    mask &= df["outlet_id"].str.contains(search_id, case=False, na=False)

fdf = df[mask].copy()

st.sidebar.caption(f"{len(fdf):,} of {len(df):,} outlets match")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🥤 Outlet Intelligence — Maximum Monthly Purchase Potential (Jan 2026)")
st.caption(
    "Transparent model: **potential = peer_ceiling × constraint_uplift × "
    "seasonality_index × spatial_multiplier**, floored at historical max, capped "
    "at peer-cluster max."
)

tab_browse, tab_outlet, tab_budget = st.tabs(
    ["📋 Browse & explore", "🔍 Outlet drill-down", "💰 Budget allocation (Western)"]
)

# ===========================================================================
# TAB 1 — Browse
# ===========================================================================
with tab_browse:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outlets", f"{len(fdf):,}")
    c2.metric("Median potential (L)", _fmt(fdf["mult_potential"].median()))
    c3.metric("Total potential (L)", _fmt(fdf["mult_potential"].sum()))
    if "hist_total_median" in fdf:
        lift = (fdf["mult_potential"] / fdf["hist_total_median"].replace(0, np.nan)).median()
        c4.metric("Median ×vs normal sales", f"{lift:,.2f}×" if pd.notna(lift) else "—")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Potential distribution")
        hist = (
            alt.Chart(fdf.dropna(subset=["mult_potential"]))
            .mark_bar()
            .encode(
                alt.X("mult_potential:Q", bin=alt.Bin(maxbins=40), title="Predicted potential (L/month)"),
                alt.Y("count():Q", title="Outlets"),
                tooltip=[alt.Tooltip("count():Q", title="Outlets")],
            )
            .properties(height=280)
        )
        st.altair_chart(hist, width="stretch")
    with right:
        st.subheader("Potential by province")
        if "province" in fdf:
            byp = (
                fdf.groupby("province")["mult_potential"]
                .agg(["count", "median", "sum"])
                .reset_index()
                .rename(columns={"count": "outlets", "median": "median_L", "sum": "total_L"})
            )
            st.dataframe(byp, width="stretch", hide_index=True)

    # Map (only outlets with valid coords)
    if {"latitude", "longitude"}.issubset(fdf.columns):
        geo = fdf.dropna(subset=["latitude", "longitude"]).rename(
            columns={"latitude": "lat", "longitude": "lon"}
        )
        if len(geo):
            st.subheader(f"Outlet map ({len(geo):,} geocoded)")
            st.map(geo[["lat", "lon"]], size=20)

    st.subheader("Outlet table")
    show_cols = [
        c for c in [
            "outlet_id", "province", "distributor_id", "outlet_type", "outlet_size",
            "mult_potential", "hist_total_median", "hist_total_max",
            "constraint_uplift", "seasonality_jan_index", "spatial_multiplier",
            "Trade_Spend_Allocation_LKR",
        ] if c in fdf.columns
    ]
    st.dataframe(
        fdf[show_cols].sort_values("mult_potential", ascending=False),
        width="stretch", hide_index=True,
    )
    st.download_button(
        "⬇ Download filtered table (CSV)",
        fdf[show_cols].to_csv(index=False).encode(),
        file_name="outlets_filtered.csv",
        mime="text/csv",
    )

# ===========================================================================
# TAB 2 — Outlet drill-down (the XAI centerpiece)
# ===========================================================================
with tab_outlet:
    if fdf.empty:
        st.warning("No outlets match the current filters.")
    else:
        options = fdf.sort_values("mult_potential", ascending=False)["outlet_id"].tolist()
        oid = st.selectbox("Select an outlet", options)
        row = df[df["outlet_id"] == oid].iloc[0]

        st.subheader(f"{oid} — {row.get('outlet_type', '?')} · {row.get('outlet_size', '?')} · "
                     f"{row.get('province', '?')} · {row.get('distributor_id', '?')}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted potential (L/mo)", _fmt(row["mult_potential"]))
        m2.metric("Normal monthly sales (L)", _fmt(row.get("hist_total_median")))
        if pd.notna(row.get("hist_total_median")) and row.get("hist_total_median"):
            m3.metric("Uplift vs normal", f"{row['mult_potential']/row['hist_total_median']:,.2f}×")
        m4.metric("Historical max seen (L)", _fmt(row.get("hist_total_max")))

        # ----- Factor decomposition (waterfall) -----
        st.markdown("### Why this score? — factor decomposition")
        pc = float(row["peer_ceiling"])
        cu = float(row["constraint_uplift"])
        si = float(row["seasonality_jan_index"])
        sm = float(row["spatial_multiplier"])
        s0 = pc
        s1 = s0 * cu
        s2 = s1 * si
        s3 = s2 * sm
        final = float(row["mult_potential"])

        steps = pd.DataFrame([
            {"stage": "1 · Peer ceiling (base)", "value": s0, "delta": s0,
             "note": "High-quantile of comparable outlets' unconstrained months"},
            {"stage": "2 · × Constraint uplift", "value": s1, "delta": s1 - s0,
             "note": f"×{cu:.3f} — left-censorship correction "
                     f"({int(row.get('months_constrained', 0))}/{int(row.get('months_observed', 0))} months constrained)"},
            {"stage": "3 · × Seasonality (Jan)", "value": s2, "delta": s2 - s1,
             "note": f"×{si:.3f} — January index for {row.get('distributor_id', '?')}"},
            {"stage": "4 · × Spatial multiplier", "value": s3, "delta": s3 - s2,
             "note": f"×{sm:.3f} — gravity accessibility vs competitive saturation"},
            {"stage": "5 · Floor/cap guardrail", "value": final, "delta": final - s3,
             "note": "Floored at historical max, capped at peer-cluster max"},
        ])

        wf = (
            alt.Chart(steps)
            .mark_bar()
            .encode(
                x=alt.X("stage:N", sort=None, title=None,
                        axis=alt.Axis(labelAngle=-20, labelLimit=200)),
                y=alt.Y("value:Q", title="Running potential (L/month)"),
                color=alt.condition(alt.datum.delta >= 0, alt.value("#2c7fb8"), alt.value("#d95f0e")),
                tooltip=["stage", alt.Tooltip("value:Q", format=",.0f"),
                         alt.Tooltip("delta:Q", format="+,.0f"), "note"],
            )
            .properties(height=300)
        )
        st.altair_chart(wf, width="stretch")

        # Guardrail note
        if abs(final - s3) > max(1.0, 0.005 * s3):
            if final > s3:
                st.info(f"🛟 **Floor was binding:** raw factor product was {s3:,.0f} L, "
                        f"lifted to the outlet's historical max of {final:,.0f} L.")
            else:
                st.info(f"🧢 **Cap was binding:** raw factor product was {s3:,.0f} L, "
                        f"capped to peer-cluster max at {final:,.0f} L (anti-blow-up guardrail).")

        st.dataframe(
            steps[["stage", "value", "delta", "note"]].rename(
                columns={"value": "running_L", "delta": "Δ_L"}),
            width="stretch", hide_index=True,
        )

        # ----- Spatial + robustness detail -----
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("#### Spatial drivers")
            st.write(f"- **Gravity accessibility:** {row.get('poi_accessibility', float('nan')):,.2f}")
            st.write(f"- **Competitive saturation:** {row.get('saturation_index', float('nan')):,.2f}")
            st.write(f"- **Our outlets within radius:** {int(row.get('own_network_density', 0))}")
            st.write(f"- **Spatial multiplier:** ×{sm:.3f} "
                     f"({'lift' if sm > 1 else 'discount' if sm < 1 else 'neutral'})")
        with d2:
            st.markdown("#### Robustness cross-check")
            ens = row.get("ensemble_potential")
            st.write(f"- **Transparent model:** {final:,.0f} L")
            st.write(f"- **Independent ensemble:** {_fmt(ens)} L")
            st.write(f"- **Divergence:** {row.get('divergence_pct', float('nan')):,.1f}% "
                     "(structural — ensemble is a censored conservative band)")
            if bool(row.get("divergence_flag", False)):
                st.warning("⚠ Flagged: divergence far exceeds the structural offset — "
                           "inspect this outlet's inputs.")

        # ----- Budget allocation (if any) -----
        if pd.notna(row.get("Trade_Spend_Allocation_LKR")) and row.get("Trade_Spend_Allocation_LKR", 0) > 0:
            st.markdown("#### Recommended trade spend")
            b1, b2, b3 = st.columns(3)
            b1.metric("Allocation (LKR)", _fmt(row["Trade_Spend_Allocation_LKR"]))
            b2.metric("Headroom (L)", _fmt(row.get("headroom")))
            b3.metric("Expected incremental (L)", _fmt(row.get("expected_incremental_liters")))

        # ----- AI explanation -----
        st.markdown("---")
        st.markdown("#### 🤖 AI Explanation")
        if not groq_api_key:
            st.caption(
                "Add a Groq API key in the sidebar to generate a plain-English "
                "explanation of this outlet's score."
            )
        else:
            cache_key = f"ai_explanation_{oid}"
            if st.button("Generate explanation", key=f"btn_{oid}"):
                with st.spinner("Generating explanation…"):
                    explanation = generate_outlet_explanation(
                        dict(row), api_key=groq_api_key
                    )
                st.session_state[cache_key] = explanation

            cached = st.session_state.get(cache_key)
            if cached:
                st.info(cached)
                st.caption("Powered by Groq · llama-3.3-70b-versatile")

# ===========================================================================
# TAB 3 — Budget allocation
# ===========================================================================
with tab_budget:
    alloc_path = config.PREDICTIONS_DIR / "galaxy_pegasus_budget_allocations.csv"
    dist_path = config.PREDICTIONS_DIR / "budget_allocation_by_distributor.csv"
    if not alloc_path.exists():
        st.info("No budget allocation found. Run "
                "`python -m src.optimization.budget_allocation`.")
    else:
        wb = df[df.get("Trade_Spend_Allocation_LKR", pd.Series(dtype=float)).fillna(0) > 0]
        total = float(config.BUDGET_CONFIG["total_budget_lkr"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Budget (LKR)", _fmt(total))
        c2.metric("Allocated (LKR)", _fmt(df["Trade_Spend_Allocation_LKR"].sum()
                                          if "Trade_Spend_Allocation_LKR" in df else np.nan))
        c3.metric("Outlets funded", f"{len(wb):,}")
        if "expected_incremental_liters" in df:
            c4.metric("Projected incremental (L)", _fmt(df["expected_incremental_liters"].sum()))

        if dist_path.exists():
            st.subheader("By distributor")
            st.dataframe(pd.read_csv(dist_path), width="stretch", hide_index=True)

        st.subheader("Funded outlets")
        cols = [c for c in ["outlet_id", "distributor_id", "outlet_type",
                            "hist_total_median", "mult_potential", "headroom",
                            "Trade_Spend_Allocation_LKR", "expected_incremental_liters"]
                if c in wb.columns]
        st.dataframe(
            wb[cols].sort_values("Trade_Spend_Allocation_LKR", ascending=False),
            width="stretch", hide_index=True,
        )
        st.caption("Greedy marginal allocation over a concave diminishing-returns "
                   "response — provably optimal for this objective.")
