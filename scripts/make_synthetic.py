"""Generate synthetic data matching the expected raw schema.

Used solely to smoke-test the full pipeline end-to-end without the real data.
The synthetic generator injects each anomaly class the DQ engine looks for
(duplicates, nulls, RI breaks, constant runs, blackouts, credit caps, OOB
coords) so we can confirm the rejection store fires correctly.

Run::

    python scripts/make_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import config


N_OUTLETS = 300
DISTRIBUTORS = config.EXPECTED_DISTRIBUTORS
PROVINCES = {
    "DIST_W_01": "Western",  "DIST_W_02": "Western",  "DIST_W_03": "Western",
    "DIST_C_01": "Central",  "DIST_C_02": "Central",  "DIST_C_03": "Central",
    "DIST_NW_01": "North-Western", "DIST_NW_02": "North-Western",
    "DIST_S_01": "Southern", "DIST_S_02": "Southern",
}
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")
rng = np.random.default_rng(42)


def make_outlets(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        dist = DISTRIBUTORS[i % len(DISTRIBUTORS)]
        rows.append({
            "outlet_id": f"OUT_{i:05d}",
            "distributor_id": dist,
            "province": PROVINCES[dist],
            "district": f"DIST_{(i // 25) % 12}",
            "latitude": float(rng.uniform(5.8, 9.8)),
            "longitude": float(rng.uniform(79.5, 81.8)),
            "outlet_type": rng.choice(["grocery", "kade", "eatery", "pharmacy"]),
        })
    df = pd.DataFrame(rows)

    # Inject anomalies the DQ engine should catch:
    # 1) duplicate outlet_id
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    # 2) out-of-bounds coordinate
    df.loc[1, "latitude"] = 50.0
    # 3) null province
    df.loc[2, "province"] = ""
    return df


def make_transactions(outlets: pd.DataFrame) -> pd.DataFrame:
    valid = outlets[outlets["latitude"].between(5.5, 10.5)]
    dates = pd.date_range(START, END, freq="D")
    rows = []
    for _, o in valid.iterrows():
        oid = o["outlet_id"]
        base = float(rng.uniform(20, 400))
        season_amp = float(rng.uniform(0.1, 0.4))
        for d in dates:
            seasonal = 1 + season_amp * np.sin(2 * np.pi * d.month / 12)
            day_vol = max(0.0, rng.normal(base * seasonal, base * 0.2))
            # cap some outlets at a credit-line round number to simulate censoring
            if oid.endswith("0") and day_vol > 200:
                day_vol = 200.0
            rows.append({
                "transaction_id": f"TXN_{oid}_{d.strftime('%Y%m%d')}",
                "outlet_id": oid,
                "distributor_id": o["distributor_id"],
                "date": d,
                "sku_id": rng.choice(["SKU_A", "SKU_B", "SKU_C"]),
                "volume_liters": round(day_vol, 2),
            })
    df = pd.DataFrame(rows)

    # Inject anomalies:
    # 1) ghost entries (constant run)
    ghost_outlet = "OUT_00007"
    mask = (df["outlet_id"] == ghost_outlet) & (df["date"].between("2024-03-01", "2024-03-21"))
    df.loc[mask, "volume_liters"] = 175.0
    # 2) distributor-wide blackout day
    df.loc[(df["distributor_id"] == "DIST_W_01") & (df["date"] == "2024-07-04"), "volume_liters"] = 0
    # 3) null volume
    df.loc[df.sample(50, random_state=1).index, "volume_liters"] = np.nan
    # 4) duplicate transaction_id
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    # 5) bad outlet RI
    bad = df.iloc[[5]].copy()
    bad["outlet_id"] = "OUT_GHOST_ZZZ"
    df = pd.concat([df, bad], ignore_index=True)

    return df


def make_seasonality() -> pd.DataFrame:
    rows = []
    for dist in DISTRIBUTORS:
        for m in range(1, 13):
            idx = 1 + 0.3 * np.sin(2 * np.pi * m / 12) + rng.normal(0, 0.05)
            rows.append({"distributor_id": dist, "month": m, "seasonality_index": round(idx, 3)})
    df = pd.DataFrame(rows)
    # inject one bad value (negative)
    df.loc[0, "seasonality_index"] = -1.0
    return df


def make_holidays() -> pd.DataFrame:
    rows = [
        {"date": "2024-02-04", "holiday_name": "Independence Day", "holiday_type": "national"},
        {"date": "2024-04-13", "holiday_name": "Sinhala/Tamil New Year", "holiday_type": "national"},
        {"date": "2024-05-23", "holiday_name": "Vesak", "holiday_type": "religious"},
        {"date": "2024-12-25", "holiday_name": "Christmas", "holiday_type": "religious"},
        {"date": "2025-04-13", "holiday_name": "Sinhala/Tamil New Year", "holiday_type": "national"},
        # bad date
        {"date": "not-a-date", "holiday_name": "Garbage", "holiday_type": "test"},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    raw_dir = config.RAW_INPUT_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    outlets = make_outlets(N_OUTLETS)
    tx = make_transactions(outlets)
    seas = make_seasonality()
    hol = make_holidays()

    outlets.to_csv(raw_dir / config.SOURCE_FILES["outlets"], index=False)
    tx.to_csv(raw_dir / config.SOURCE_FILES["transactions"], index=False)
    seas.to_csv(raw_dir / config.SOURCE_FILES["seasonality"], index=False)
    hol.to_csv(raw_dir / config.SOURCE_FILES["holidays"], index=False)
    print(f"wrote synthetic CSVs under {raw_dir}")
    print(f"  outlets:      {len(outlets):,}")
    print(f"  transactions: {len(tx):,}")
    print(f"  seasonality:  {len(seas):,}")
    print(f"  holidays:     {len(hol):,}")


if __name__ == "__main__":
    main()
