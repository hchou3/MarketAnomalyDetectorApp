"""Rebuild and compare the 8 stress_extra.csv columns from the database.

Core logic (offline-testable):
  align_cboe  -- last obs_date ≤ tuesday, blank before the series starts
  align_fred  -- latest known_at ≤ tuesday (point-in-time correctness)
  compare_column -- matched / differing / missing counts and max_abs_diff

DB layer (used by the CLI):
  get_ews_tuesdays -- distinct bloomberg_ews obs_dates from the DB
  build_stress_db  -- fetch prices, align, return a wide DataFrame
  run_parity       -- orchestrate and return (results, exit_code)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

_TOLERANCE = 1e-6

# (csv_column, instrument_id, primary_provider, is_revised)
_SERIES: list[tuple[str, str, str, bool]] = [
    ("VIX3M", "vix3m", "cboe", False),
    ("VIX9D", "vix9d", "cboe", False),
    ("VVIX", "vvix", "cboe", False),
    ("SKEW", "skew", "cboe", False),
    ("OVX", "ovx", "cboe", False),
    ("GVZ", "gvz", "cboe", False),
    ("NFCI", "nfci", "fred", True),
    ("STLFSI", "stlfsi", "fred", True),
]

_CBOE_FALLBACK = "yahoo"


@dataclass
class ColumnResult:
    col: str
    is_revised: bool
    matched: int
    differing: int
    missing: int
    max_abs_diff: Optional[float]
    fails: bool


def align_cboe(prices_df: pd.DataFrame, tuesdays: list[date]) -> pd.Series:
    """Last obs_date ≤ tuesday, blank before the series starts.

    Replicates the ffill logic in fetch_stress_data.py with no carry-forward
    limit (Cboe daily closes may skip holidays, so we always use the prior day).
    """
    if prices_df.empty or tuesdays == []:
        return pd.Series(float("nan"), index=tuesdays, dtype=float)

    prices = prices_df.sort_values("obs_date").reset_index(drop=True)
    prices["_dt"] = pd.to_datetime(prices["obs_date"])
    tues_dt = pd.to_datetime(tuesdays)

    merged = pd.merge_asof(
        pd.DataFrame({"_dt": tues_dt}),
        prices[["_dt", "value"]],
        on="_dt",
        direction="backward",
    )
    result = merged["value"].to_numpy(dtype=float)
    result[tues_dt < prices["_dt"].iloc[0]] = float("nan")
    return pd.Series(result, index=tuesdays, dtype=float)


def align_fred(prices_df: pd.DataFrame, tuesdays: list[date]) -> pd.Series:
    """Latest value whose known_at ≤ tuesday.

    FRED weekly series are Friday-dated; known_at = obs_date + publication_lag_days
    (11 by default).  This ensures we never use a value before it was published.
    """
    if prices_df.empty or tuesdays == []:
        return pd.Series(float("nan"), index=tuesdays, dtype=float)

    prices = prices_df.copy()
    prices["_known_dt"] = pd.to_datetime(prices["known_at"])
    prices = prices.sort_values("_known_dt").reset_index(drop=True)

    merged = pd.merge_asof(
        pd.DataFrame({"_known_dt": pd.to_datetime(tuesdays)}),
        prices[["_known_dt", "value"]],
        on="_known_dt",
        direction="backward",
    )
    return pd.Series(merged["value"].to_numpy(dtype=float), index=tuesdays, dtype=float)


def compare_column(
    db: pd.Series,
    csv: pd.Series,
    col: str,
    is_revised: bool,
) -> ColumnResult:
    """Per-column comparison of DB-aligned vs CSV values."""
    both = pd.DataFrame({"db": db.astype(float), "csv": csv.astype(float)})
    csv_valid = both.dropna(subset=["csv"])
    missing = int(csv_valid["db"].isna().sum())
    common = csv_valid.dropna(subset=["db"])

    if common.empty:
        return ColumnResult(
            col=col,
            is_revised=is_revised,
            matched=0,
            differing=0,
            missing=missing,
            max_abs_diff=None,
            fails=not is_revised and missing > 0,
        )

    diff = (common["db"] - common["csv"]).abs()
    differing = int((diff > _TOLERANCE).sum())
    matched = len(common) - differing
    max_diff = float(diff.max())
    fails = not is_revised and (differing > 0 or missing > 0)
    return ColumnResult(
        col=col,
        is_revised=is_revised,
        matched=matched,
        differing=differing,
        missing=missing,
        max_abs_diff=max_diff,
        fails=fails,
    )


def format_report(results: list[ColumnResult], n_overlap: int, csv_path: Path) -> str:
    """Format a human-readable parity report."""
    lines = [
        f"parity-stress: comparing DB vs {csv_path} ({n_overlap} overlapping weeks)",
        "",
        f"{'Column':<8}  {'Matched':>8}  {'Differing':>9}  {'Missing':>7}"
        f"  {'MaxAbsDiff':>12}  Status",
        "-" * 66,
    ]
    for r in results:
        max_str = f"{r.max_abs_diff:.6g}" if r.max_abs_diff is not None else "—"
        if r.is_revised:
            status = "REVISED (differences not counted as failures)"
        elif r.fails:
            status = "FAIL"
        else:
            status = "OK"
        lines.append(
            f"{r.col:<8}  {r.matched:>8}  {r.differing:>9}  {r.missing:>7}  {max_str:>12}  {status}"
        )
    lines.append("")
    fails = [r for r in results if r.fails]
    if fails:
        lines.append(
            f"FAILED: {len(fails)} non-revised column(s) have mismatches or missing weeks."
        )
    else:
        revised_with_diff = [r for r in results if r.is_revised and r.differing > 0]
        if revised_with_diff:
            names = ", ".join(r.col for r in revised_with_diff)
            lines.append(
                f"OK (non-revised series match). "
                f"Revised series with differences (expected): {names}."
            )
        else:
            lines.append("OK (all series match within tolerance).")
    return "\n".join(lines)


# ── DB layer ──────────────────────────────────────────────────────────────────


def get_ews_tuesdays(session) -> list[date]:
    """Return all bloomberg_ews obs_dates from the database, sorted."""
    from sqlalchemy import distinct, select

    from .db.models import PriceRow

    rows = session.execute(
        select(distinct(PriceRow.obs_date))
        .where(PriceRow.provider == "bloomberg_ews")
        .order_by(PriceRow.obs_date)
    ).fetchall()
    return [r[0] for r in rows]


def build_stress_db(session, tuesdays: list[date]) -> pd.DataFrame:
    """Query DB and align all 8 stress series to the given Tuesday calendar."""
    from sqlalchemy import select

    from .db.models import PriceRow

    inst_ids = list({s[1] for s in _SERIES})
    rows = session.execute(
        select(
            PriceRow.instrument_id,
            PriceRow.provider,
            PriceRow.obs_date,
            PriceRow.value,
            PriceRow.known_at,
        )
        .where(PriceRow.instrument_id.in_(inst_ids))
        .order_by(PriceRow.instrument_id, PriceRow.provider, PriceRow.obs_date)
    ).fetchall()

    price_data: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row.instrument_id, row.provider)
        price_data.setdefault(key, []).append(
            {
                "obs_date": row.obs_date,
                "value": float(row.value),
                "known_at": row.known_at,
            }
        )

    def _df(key: tuple[str, str]) -> pd.DataFrame:
        data = price_data.get(key, [])
        return (
            pd.DataFrame(data) if data else pd.DataFrame(columns=["obs_date", "value", "known_at"])
        )

    out: dict[str, pd.Series] = {}
    for col, inst_id, primary_provider, _ in _SERIES:
        if primary_provider == "cboe":
            df = _df((inst_id, "cboe"))
            if df.empty:
                df = _df((inst_id, _CBOE_FALLBACK))
            out[col] = align_cboe(df, tuesdays)
        else:
            out[col] = align_fred(_df((inst_id, "fred")), tuesdays)

    return pd.DataFrame(out, index=tuesdays)


def run_parity(session, csv_path: Path) -> tuple[list[ColumnResult], int, int]:
    """Run the full parity check; return (results, exit_code, n_overlap)."""
    tuesdays = get_ews_tuesdays(session)

    csv_df = pd.read_csv(csv_path, parse_dates=["Data"])
    csv_df["Data"] = csv_df["Data"].dt.date
    csv_df = csv_df.set_index("Data")

    if not tuesdays:
        tuesdays = sorted(csv_df.index.tolist())

    db_df = build_stress_db(session, tuesdays)

    overlap = sorted(set(tuesdays) & set(csv_df.index))
    n_overlap = len(overlap)

    results: list[ColumnResult] = []
    for col, _, _, is_revised in _SERIES:
        db_col = (
            db_df[col].reindex(overlap)
            if col in db_df.columns
            else pd.Series(float("nan"), index=overlap, dtype=float)
        )
        csv_col = (
            csv_df[col].reindex(overlap).astype(float)
            if col in csv_df.columns
            else pd.Series(float("nan"), index=overlap, dtype=float)
        )
        results.append(compare_column(db_col, csv_col, col, is_revised))

    exit_code = 1 if any(r.fails for r in results) else 0
    return results, exit_code, n_overlap
