"""LLM explainability layer using Groq API.

Generates a plain-English business narrative for an outlet's potential score
by sending the factor decomposition to a Groq-hosted LLM.
"""

from __future__ import annotations

import os
from typing import Any

_SYSTEM_PROMPT = """\
You are an expert sales analytics consultant for a beverage distributor in Sri Lanka.
You explain outlet-level sales potential predictions to non-technical business users
(sales managers, territory executives) in clear, jargon-free language.

The predictions come from a transparent model:
  potential = peer_ceiling × constraint_uplift × seasonality_index × spatial_multiplier

Definitions:
- peer_ceiling: the high-quantile (p90) monthly volume of comparable outlets in the
  same cluster — the demand ceiling observable among similar unconstrained outlets.
- constraint_uplift: a multiplier ≥ 1.0 that corrects for left-censored historical
  sales (credit caps, stockouts, route limits). Higher = more constrained in the past.
- seasonality_jan_index: January seasonality for the outlet's distributor
  (1.15 = favorable, 1.00 = neutral, 0.85 = unfavorable).
- spatial_multiplier: a location quality factor based on nearby footfall-generating
  POIs (markets, transit, tourism) weighted by distance decay, discounted for
  competitive saturation. Bounded [0.70, 1.40]; 1.0 = average location.
- The final potential is floored at the outlet's own historical maximum (can't be
  below demonstrated sales) and capped at the peer-cluster maximum (anti-blow-up).

Your job: write 3–5 sentences explaining WHY this specific outlet has this potential
score, in a way that helps a sales manager decide whether to prioritize it and how
much trade spend to allocate. Be concrete — reference the actual numbers. Do not
mention "the model" or technical jargon. Do not start with "This outlet".
"""


def generate_outlet_explanation(outlet_data: dict[str, Any], api_key: str | None = None) -> str | None:
    """Return a natural-language explanation for an outlet's potential score.

    Args:
        outlet_data: Dict with outlet fields (factors, historical stats, spatial, budget).
        api_key: Groq API key. Falls back to GROQ_API_KEY env var.

    Returns:
        Explanation string, or None if the API key is missing / call fails.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return None

    try:
        from groq import Groq
    except ImportError:
        return None

    def _fmt(v: Any, unit: str = "", decimals: int = 0) -> str:
        if v is None:
            return "N/A"
        try:
            f = float(v)
            if decimals:
                return f"{f:,.{decimals}f}{unit}"
            return f"{f:,.0f}{unit}"
        except (TypeError, ValueError):
            return str(v)

    oid = outlet_data.get("outlet_id", "Unknown")
    o_type = outlet_data.get("outlet_type", "Unknown")
    o_size = outlet_data.get("outlet_size", "Unknown")
    province = outlet_data.get("province", "Unknown")
    dist_id = outlet_data.get("distributor_id", "Unknown")

    potential = outlet_data.get("mult_potential")
    peer_ceil = outlet_data.get("peer_ceiling")
    uplift = outlet_data.get("constraint_uplift")
    seasonality = outlet_data.get("seasonality_jan_index")
    spatial = outlet_data.get("spatial_multiplier")
    hist_med = outlet_data.get("hist_total_median")
    hist_max = outlet_data.get("hist_total_max")
    months_obs = outlet_data.get("months_observed", 0)
    months_con = outlet_data.get("months_constrained", 0)
    poi_acc = outlet_data.get("poi_accessibility")
    sat_idx = outlet_data.get("saturation_index")
    budget = outlet_data.get("Trade_Spend_Allocation_LKR")
    headroom = outlet_data.get("headroom")
    incremental = outlet_data.get("expected_incremental_liters")

    constraint_pct = (int(months_con) / int(months_obs) * 100) if months_obs else 0

    user_message = f"""Outlet: {oid}
Type/Size: {o_type} / {o_size}
Province: {province} | Distributor: {dist_id}

Factor decomposition:
- Peer ceiling (base):       {_fmt(peer_ceil)} L/month
- × Constraint uplift:       {_fmt(uplift, decimals=3)}×  ({constraint_pct:.0f}% of months were constrained)
- × Seasonality (January):   {_fmt(seasonality, decimals=2)}×
- × Spatial multiplier:      {_fmt(spatial, decimals=3)}×

Final predicted potential:   {_fmt(potential)} L/month
Normal monthly sales:        {_fmt(hist_med)} L  (historical median)
Historical maximum ever:     {_fmt(hist_max)} L
POI accessibility score:     {_fmt(poi_acc, decimals=2)}
Competitive saturation:      {_fmt(sat_idx, decimals=2)}

Budget allocation:           {_fmt(budget, unit=' LKR') if budget else 'None'}
Available headroom:          {_fmt(headroom)} L
Expected incremental lift:   {_fmt(incremental)} L

Write a 3–5 sentence explanation for the sales manager."""

    client = Groq(api_key=key)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None
