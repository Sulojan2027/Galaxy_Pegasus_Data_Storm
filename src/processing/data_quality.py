"""Reusable, parameterizable Data Quality (DQ) check engine."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CheckFn = Callable[..., tuple[pd.DataFrame, pd.DataFrame]]

# Registry: name -> function. Allows config-driven dispatch.
_REGISTRY: dict[str, CheckFn] = {}


def register(name: str) -> Callable[[CheckFn], CheckFn]:
    def _wrap(fn: CheckFn) -> CheckFn:
        _REGISTRY[name] = fn
        return fn
    return _wrap


# Helpers
def _tag_reason(df: pd.DataFrame, reason: str) -> pd.DataFrame:
    """Attach (or append to) a `failure_reason` column."""
    if df.empty:
        out = df.copy()
        out["failure_reason"] = pd.Series(dtype="object")
        return out
    out = df.copy()
    if "failure_reason" in out.columns:
        out["failure_reason"] = out["failure_reason"].fillna("").astype(str)
        out["failure_reason"] = np.where(
            out["failure_reason"].eq(""),
            reason,
            out["failure_reason"] + " | " + reason,
        )
    else:
        out["failure_reason"] = reason
    return out


# Core checks
@register("duplicate")
def check_duplicates(
    df: pd.DataFrame,
    keys: list[str],
    keep: str | bool = "first",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect duplicate rows on a configurable composite primary key."""
    if not keys:
        return df, _tag_reason(df.iloc[0:0], "duplicate:no_keys")

    missing = [k for k in keys if k not in df.columns]
    if missing:
        logger.warning("duplicate check skipped, missing keys: %s", missing)
        return df, _tag_reason(df.iloc[0:0], "duplicate:missing_keys")

    dup_mask = df.duplicated(subset=keys, keep=keep)
    rejected = _tag_reason(df[dup_mask], f"duplicate_on:{'+'.join(keys)}")
    passing = df[~dup_mask]
    return passing, rejected


@register("null")
def check_nulls(
    df: pd.DataFrame,
    mandatory_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag rows where any mandatory column is null OR empty-string."""
    cols = [c for c in mandatory_cols if c in df.columns]
    if not cols:
        return df, _tag_reason(df.iloc[0:0], "null:no_cols")

    null_mask = pd.Series(False, index=df.index)
    reasons: dict[int, list[str]] = {}
    for c in cols:
        s = df[c]
        col_null = s.isna() | (s.astype(str).str.strip() == "")
        null_mask |= col_null
        for idx in df.index[col_null]:
            reasons.setdefault(idx, []).append(c)

    rejected_rows = df[null_mask].copy()
    rejected_rows["failure_reason"] = [
        "null_in:" + ",".join(reasons.get(idx, []))
        for idx in rejected_rows.index
    ]
    passing = df[~null_mask]
    return passing, rejected_rows


@register("referential_integrity")
def check_referential_integrity(
    df: pd.DataFrame,
    fk_col: str,
    ref_df: pd.DataFrame,
    ref_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate that every fk value exists in the reference column."""
    if fk_col not in df.columns or ref_col not in ref_df.columns:
        logger.warning("RI check skipped (%s vs %s.%s)", fk_col, ref_df, ref_col)
        return df, _tag_reason(df.iloc[0:0], "ri:missing_cols")
    valid_set = set(ref_df[ref_col].dropna().unique())
    mask_bad = ~df[fk_col].isin(valid_set)
    rejected = _tag_reason(df[mask_bad], f"ri_violation:{fk_col}")
    return df[~mask_bad], rejected


@register("value_range")
def check_value_range(
    df: pd.DataFrame,
    col: str,
    min_: float | None = None,
    max_: float | None = None,
    inclusive: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numeric range check on a single column."""
    if col not in df.columns:
        return df, _tag_reason(df.iloc[0:0], f"range:missing_col:{col}")

    numeric = pd.to_numeric(df[col], errors="coerce")
    bad = pd.Series(False, index=df.index)
    if min_ is not None:
        bad |= numeric.lt(min_) if inclusive else numeric.le(min_)
    if max_ is not None:
        bad |= numeric.gt(max_) if inclusive else numeric.ge(max_)
    # Flag non-numeric rows on numeric columns
    bad |= numeric.isna() & df[col].notna()

    rejected = _tag_reason(df[bad], f"range_violation:{col}[{min_},{max_}]")
    return df[~bad], rejected


@register("format")
def check_format(
    df: pd.DataFrame,
    col: str,
    dtype: str | None = None,
    regex: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate that a column conforms to an expected type/format."""
    if col not in df.columns:
        return df, _tag_reason(df.iloc[0:0], f"format:missing_col:{col}")

    bad = pd.Series(False, index=df.index)

    if dtype in ("int", "float"):
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad |= coerced.isna() & df[col].notna()
        if dtype == "int":
            bad |= (coerced.notna()) & (coerced.astype("Float64") % 1 != 0)
    elif dtype in ("date", "datetime"):
        coerced = pd.to_datetime(df[col], errors="coerce", utc=False)
        bad |= coerced.isna() & df[col].notna()
    elif dtype == "string":
        bad |= ~df[col].apply(lambda v: isinstance(v, str) or pd.isna(v))

    if regex:
        pat = re.compile(regex)
        bad |= ~df[col].fillna("").astype(str).apply(lambda v: bool(pat.fullmatch(v)))

    rejected = _tag_reason(df[bad], f"format_violation:{col}({dtype or ''}|{regex or ''})")
    return df[~bad], rejected


# Custom checks
@register("coord_swap_fix")
def check_coord_swap(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    lat_bounds: tuple[float, float] = (5.5, 10.5),
    lon_bounds: tuple[float, float] = (79.0, 82.5),
    autofix: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect and optionally auto-fix swapped Latitude and Longitude rows."""
    if not {lat_col, lon_col}.issubset(df.columns):
        return df, _tag_reason(df.iloc[0:0], "coord_swap:missing_cols")

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")
    swapped = (
        lat.between(*lon_bounds) & lon.between(*lat_bounds)
        & ~(lat.between(*lat_bounds) & lon.between(*lon_bounds))
    )
    if autofix:
        out = df.copy()
        out.loc[swapped, lat_col] = lon[swapped].values
        out.loc[swapped, lon_col] = lat[swapped].values
        n_fixed = int(swapped.sum())
        if n_fixed:
            logger.info("coord_swap: auto-fixed %d swapped lat/lon rows", n_fixed)
        empty_rej = _tag_reason(df.iloc[0:0], f"coord_swap:autofixed n={n_fixed}")
        return out, empty_rej
    rejected = _tag_reason(df[swapped], "coord_swap")
    return df[~swapped], rejected


@register("text_normalize")
def check_text_normalize(
    df: pd.DataFrame,
    col: str,
    mapping: dict[str, str],
    case_insensitive: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trim and map common typos to a canonical value."""
    if col not in df.columns:
        return df, _tag_reason(df.iloc[0:0], f"text_normalize:missing_col:{col}")

    out = df.copy()
    s = out[col].astype("string").str.strip()
    if case_insensitive:
        key = s.str.lower()
    else:
        key = s
    mapped = key.map(mapping)
    out[col] = mapped.fillna(s)
    empty_rej = _tag_reason(df.iloc[0:0], f"text_normalize:{col}")
    return out, empty_rej


@register("negative_volume_tag")
def check_negative_volume_tag(
    df: pd.DataFrame,
    value_col: str = "volume_liters",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tag rows with negative volume as returns/credits."""
    if value_col not in df.columns:
        return df, _tag_reason(df.iloc[0:0], "neg_vol:missing_col")
    out = df.copy()
    out["_is_return"] = pd.to_numeric(out[value_col], errors="coerce").lt(0).fillna(False)
    return out, _tag_reason(df.iloc[0:0], "neg_vol:tagged")


@register("low_volume_month_tag")
def check_low_volume_month(
    df: pd.DataFrame,
    group_keys: list[str],
    value_col: str = "volume_liters",
    quantile: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tag low-volume months as a stockout proxy."""
    if value_col not in df.columns or not all(c in df.columns for c in group_keys):
        return df, _tag_reason(df.iloc[0:0], "low_vol:missing_cols")
    out = df.copy()
    out["_v"] = pd.to_numeric(out[value_col], errors="coerce")
    thresh = out.groupby(group_keys)["_v"].transform(lambda s: s.quantile(quantile))
    out["_low_volume_month_flag"] = (out["_v"] <= thresh).fillna(False) & out["_v"].notna()
    out = out.drop(columns=["_v"])
    return out, _tag_reason(df.iloc[0:0], "low_vol:tagged")


@register("credit_cap_signature")
def check_credit_cap_signature(
    df: pd.DataFrame,
    group_keys: list[str],
    value_col: str,
    modulos: list[int] | None = None,
    min_hits: int = 5,
    min_share: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Heuristic detector of credit-cap fingerprints."""
    modulos = modulos or [100, 500, 1000]
    if value_col not in df.columns:
        return df, _tag_reason(df.iloc[0:0], "credit_cap:missing_col")

    d = df.copy()
    d["_v"] = pd.to_numeric(d[value_col], errors="coerce")
    d["_round"] = False
    for m in modulos:
        d["_round"] |= (d["_v"] % m == 0) & (d["_v"] > 0)
    grp = d.groupby(group_keys, dropna=False)
    share = grp["_round"].transform("mean")
    hits = grp["_round"].transform("sum")
    suspect = (share >= min_share) & (hits >= min_hits) & d["_round"]

    # This check TAGS, it doesn't reject. We add `_credit_cap_flag`.
    out = df.copy()
    out["_credit_cap_flag"] = suspect.values
    empty_rej = _tag_reason(df.iloc[0:0], "credit_cap:tag_only")
    return out, empty_rej


# Pipeline driver
@dataclass
class CheckSpec:
    """One step in a check pipeline."""
    name: str                                # registered check name
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    dataset: str
    passing: pd.DataFrame
    rejected: pd.DataFrame
    summary: list[dict[str, Any]]


def run_checks(
    df: pd.DataFrame,
    dataset_name: str,
    pipeline: list[CheckSpec],
) -> CheckResult:
    """Run a sequence of checks on `df`, accumulating rejections."""
    passing = df
    all_rejected = []
    summary: list[dict[str, Any]] = []

    for step in pipeline:
        fn = _REGISTRY.get(step.name)
        if fn is None:
            logger.warning("unknown check '%s' on dataset '%s'", step.name, dataset_name)
            continue
        before = len(passing)
        passing, rejected = fn(passing, **step.params)
        after = len(passing)
        rejected_n = len(rejected)
        if rejected_n:
            rejected = rejected.assign(_check=step.name, _dataset=dataset_name)
            all_rejected.append(rejected)
        summary.append({
            "dataset": dataset_name,
            "check": step.name,
            "params": step.params,
            "rows_before": before,
            "rows_rejected": rejected_n,
            "rows_after": after,
        })
        logger.info(
            "[%s] %s: %d -> %d (rejected %d)",
            dataset_name, step.name, before, after, rejected_n,
        )

    rejected_df = (
        pd.concat(all_rejected, ignore_index=True)
        if all_rejected
        else pd.DataFrame(columns=list(df.columns) + ["failure_reason", "_check", "_dataset"])
    )
    return CheckResult(
        dataset=dataset_name,
        passing=passing.reset_index(drop=True),
        rejected=rejected_df,
        summary=summary,
    )


def available_checks() -> list[str]:
    """Return registered check names — useful for introspection / tests."""
    return sorted(_REGISTRY.keys())
