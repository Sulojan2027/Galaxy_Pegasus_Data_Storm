"""Reusable, parameterizable Data Quality (DQ) check engine.

Design contract — every check function returns a tuple::

    (passing_df, rejected_df_with_failure_reason)

Each rejected row carries a `failure_reason` column describing exactly which
check fired. Rejected rows are NEVER silently dropped — Silver layer writes
them to ``data/silver/_rejected/`` for audit per the brief.

A YAML/dict-driven runner (`run_checks`) applies a *pipeline* of checks to a
DataFrame, accumulating rejections.  Adding a new check is a one-function
addition + a config entry — no orchestration code changes required.
"""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Core checks (the five mandated by the brief)
# ---------------------------------------------------------------------------
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
    # also flag non-numeric rows on numeric columns
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
    """Validate that a column conforms to an expected type/format.

    `dtype` may be one of: "int", "float", "date", "datetime", "string".
    `regex`, if supplied, is applied after type coercion succeeds.
    """
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


# ---------------------------------------------------------------------------
# Custom checks for FMCG / legacy SFA artifacts
# ---------------------------------------------------------------------------
@register("constant_run")
def check_constant_runs(
    df: pd.DataFrame,
    group_keys: list[str],
    order_col: str,
    value_col: str,
    min_run: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag suspicious runs of identical non-zero values per group.

    Catches "ghost entries" where the SFA app pre-fills the same volume across
    consecutive days for the same outlet — a classic data-entry shortcut.
    """
    needed = set(group_keys + [order_col, value_col])
    if not needed.issubset(df.columns):
        return df, _tag_reason(df.iloc[0:0], "constant_run:missing_cols")

    d = df.sort_values(group_keys + [order_col]).copy()
    d["_v"] = pd.to_numeric(d[value_col], errors="coerce")
    grp = d.groupby(group_keys, sort=False)
    # mark rows where value equals previous value within group
    d["_same"] = grp["_v"].transform(lambda s: s.eq(s.shift(1)))
    # consecutive run id (resets when _same flips False)
    d["_run_id"] = (~d["_same"].fillna(False)).cumsum()
    d["_run_len"] = d.groupby("_run_id")["_v"].transform("size")
    suspicious_mask = (
        (d["_run_len"] >= min_run) & d["_v"].fillna(0).gt(0)
    )
    rejected_idx = d.index[suspicious_mask]
    rejected = _tag_reason(
        df.loc[rejected_idx],
        f"constant_run>={min_run}:{value_col}",
    )
    passing = df.drop(index=rejected_idx)
    return passing, rejected


@register("distributor_blackout")
def check_distributor_blackout(
    df: pd.DataFrame,
    distributor_col: str,
    date_col: str,
    outlet_col: str,
    value_col: str,
    blackout_threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect days where ~all outlets of a distributor record zero volume.

    Almost always a connectivity blackout, route truck breakdown, or SFA sync
    failure — not real demand collapse. We quarantine the affected rows so
    they don't poison the "active days" denominator downstream.
    """
    needed = {distributor_col, date_col, outlet_col, value_col}
    if not needed.issubset(df.columns):
        return df, _tag_reason(df.iloc[0:0], "blackout:missing_cols")

    d = df.copy()
    d["_v"] = pd.to_numeric(d[value_col], errors="coerce").fillna(0)
    # outlets-per-(distributor,date)
    grp = d.groupby([distributor_col, date_col], dropna=False)
    activity = grp.agg(
        zero_rate=("_v", lambda s: float((s == 0).mean())),
        outlets=(outlet_col, "nunique"),
    ).reset_index()
    bad_days = activity.loc[
        (activity["zero_rate"] >= blackout_threshold) & (activity["outlets"] >= 3),
        [distributor_col, date_col],
    ]
    if bad_days.empty:
        return df, _tag_reason(df.iloc[0:0], "blackout:none")

    bad_keys = set(map(tuple, bad_days.itertuples(index=False, name=None)))
    mask = df.apply(
        lambda r: (r[distributor_col], r[date_col]) in bad_keys, axis=1
    )
    rejected = _tag_reason(
        df[mask],
        f"distributor_blackout(zero>={blackout_threshold:.0%})",
    )
    return df[~mask], rejected


@register("credit_cap_signature")
def check_credit_cap_signature(
    df: pd.DataFrame,
    group_keys: list[str],
    value_col: str,
    modulos: list[int] | None = None,
    min_hits: int = 5,
    min_share: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Heuristic detector of credit-cap fingerprints.

    An outlet whose recurring volume is a clean multiple of 100/500/1000 across
    many transactions is almost certainly hitting a credit limit, not its true
    demand ceiling. We DO NOT drop these rows — they are valuable signal — but
    we *tag* them so the modeling layer can mark these months as "constrained"
    (i.e., not eligible as ground-truth ceilings).
    """
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

    # IMPORTANT: this check TAGS, it doesn't reject. We return all rows as
    # passing but augment with a `_credit_cap_flag` column for downstream use.
    out = df.copy()
    out["_credit_cap_flag"] = suspect.values
    empty_rej = _tag_reason(df.iloc[0:0], "credit_cap:tag_only")
    return out, empty_rej


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------
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
    """Run a sequence of checks on `df`, accumulating rejections.

    Each check sees the *current passing set* — so later checks don't waste
    time on rows already rejected. The rejected store is the union of all
    quarantined rows with their reason.
    """
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
