"""FX / equity timing diagnostic.

Computes weekly-change correlation between a set of instruments and their
bloomberg_ews reference at obs_date shifts of -1, 0, and +1 business day.
A positive shift moves live observation dates forward by N business days
before alignment to EWS Tuesdays.

Usage
-----
::

    from market_anomaly_detector.timing import compute_timing_diagnostic, write_timing_report

    results = compute_timing_diagnostic(session, instrument_ids=["gbp_usd", "usd_jpy"])
    write_timing_report(results, Path("docs/reports/2026-10-22-timing-diagnostic.md"))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_SHIFTS = (-1, 0, 1)


@dataclass
class TimingRow:
    instrument_id: str
    provider: str
    symbol: str
    reference_instrument: Optional[str]
    ews_instrument_id: str
    corr_minus1: Optional[float]
    corr_0: Optional[float]
    corr_plus1: Optional[float]
    best_shift: int
    best_corr: Optional[float]
    note: str


def compute_timing_diagnostic(
    session,
    instrument_ids: list[str],
) -> list[TimingRow]:
    """Compute shift-correlation table for *instrument_ids*.

    For each instrument the live series' obs_dates are shifted by -1, 0, +1
    business days (np.busday_offset) before alignment to EWS Tuesdays via
    backward-fill merge_asof.
    """
    from sqlalchemy import select

    from .coverage import _align_weekly, _compute_changes, _correlation
    from .db.models import PriceRow

    rows = session.execute(
        select(
            PriceRow.instrument_id,
            PriceRow.provider,
            PriceRow.obs_date,
            PriceRow.value,
        ).order_by(PriceRow.instrument_id, PriceRow.provider, PriceRow.obs_date)
    ).fetchall()

    prices: dict[tuple[str, str], pd.DataFrame] = {}
    for r in rows:
        key = (r.instrument_id, r.provider)
        if key not in prices:
            prices[key] = []
        prices[key].append({"obs_date": r.obs_date, "value": float(r.value)})
    prices = {k: pd.DataFrame(v) for k, v in prices.items()}

    from .registry import load_registry

    instruments = {inst.id: inst for inst in load_registry()}

    results: list[TimingRow] = []
    for inst_id in instrument_ids:
        inst = instruments.get(inst_id)
        if inst is None:
            log.warning("timing-diagnostic: instrument %s not found in registry", inst_id)
            continue

        live_sources = sorted(
            [s for s in inst.sources if s.provider != "bloomberg_ews" and s.symbol],
            key=lambda s: s.priority,
        )
        chosen_src = None
        chosen_df: Optional[pd.DataFrame] = None
        for src in live_sources:
            df = prices.get((inst_id, src.provider))
            if df is not None and not df.empty:
                chosen_src = src
                chosen_df = df
                break

        if chosen_src is None or chosen_df is None:
            log.warning("timing-diagnostic: no live data for %s", inst_id)
            continue

        ews_inst_id = inst.reference_instrument if inst.reference_instrument else inst.id
        ews_df = prices.get((ews_inst_id, "bloomberg_ews"))
        if ews_df is None or ews_df.empty:
            log.warning(
                "timing-diagnostic: no EWS data for %s (reference %s)", inst_id, ews_inst_id
            )
            continue

        units = inst.units.value
        corrs: dict[int, Optional[float]] = {}
        for shift in _SHIFTS:
            shifted_df = _shift_obs_dates(chosen_df, shift)
            aligned = _align_weekly(shifted_df, ews_df)
            if len(aligned) < 4:
                corrs[shift] = None
                continue
            live_chg = _compute_changes(aligned["live_value"].reset_index(drop=True), units)
            ews_chg = _compute_changes(aligned["ews_value"].reset_index(drop=True), units)
            corrs[shift] = _correlation(live_chg, ews_chg)

        valid = {s: c for s, c in corrs.items() if c is not None}
        if valid:
            best_shift = max(valid, key=lambda s: valid[s])
            best_corr = valid[best_shift]
        else:
            best_shift = 0
            best_corr = None

        note = _shift_note(corrs, inst_id)

        results.append(
            TimingRow(
                instrument_id=inst_id,
                provider=chosen_src.provider,
                symbol=chosen_src.symbol,
                reference_instrument=inst.reference_instrument,
                ews_instrument_id=ews_inst_id,
                corr_minus1=corrs.get(-1),
                corr_0=corrs.get(0),
                corr_plus1=corrs.get(1),
                best_shift=best_shift,
                best_corr=best_corr,
                note=note,
            )
        )

    return results


def write_timing_report(results: list[TimingRow], path: Path) -> None:
    """Write markdown timing-diagnostic report to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = _render_timing_markdown(results)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Internal helpers ──────────────────────────────────────────────────────────


def _shift_obs_dates(df: pd.DataFrame, shift: int) -> pd.DataFrame:
    """Shift obs_date by *shift* business days (positive = forward)."""
    if shift == 0:
        return df
    out = df.copy()
    out["obs_date"] = out["obs_date"].apply(
        lambda d: np.busday_offset(d.isoformat(), shift).astype(date)
    )
    return out


def _shift_note(corrs: dict[int, Optional[float]], inst_id: str) -> str:
    c0 = corrs.get(0)
    best = max(
        (s for s in corrs if corrs[s] is not None),
        key=lambda s: corrs[s] or -99,
        default=0,
    )
    best_c = corrs.get(best)
    if best_c is None:
        return "insufficient data"
    if best == 0:
        return "no shift improves correlation"
    direction = "+1 bd" if best == 1 else "-1 bd"
    improvement = ""
    if c0 is not None and best_c is not None:
        improvement = f"; improves corr from {c0:.3f} to {best_c:.3f}"
    if best_c >= 0.98:
        return f"{direction} raises corr to exact-threshold{improvement}"
    elif best_c >= 0.80:
        return f"{direction} raises corr to proxy-threshold{improvement}"
    else:
        return f"{direction} is best shift but corr still below proxy{improvement}"


def _render_timing_markdown(results: list[TimingRow]) -> list[str]:
    today = date.today()
    lines: list[str] = [
        "# FX / Equity Timing Diagnostic",
        "",
        f"*Generated {today}. Shifts applied to live obs_dates before aligning to EWS Tuesdays.*",
        "",
        "A **positive shift (+1 bd)** moves the live date forward (e.g., a Thursday close",
        "treated as Friday), which may align better with an EWS snapshot taken at a later",
        "point in the week.",
        "",
        "## Correlation at obs_date Shifts",
        "",
        "| Instrument | Provider | Symbol | EWS Reference | corr(−1 bd) | corr(0) | corr(+1 bd)"
        " | Best Shift | Best Corr | Note |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.3f}" if v is not None else "—"

    for r in results:
        lines.append(
            f"| {r.instrument_id} | {r.provider} | {r.symbol} | {r.ews_instrument_id}"
            f" | {_fmt(r.corr_minus1)} | {_fmt(r.corr_0)} | {_fmt(r.corr_plus1)}"
            f" | {r.best_shift:+d} | {_fmt(r.best_corr)} | {r.note} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- **corr ≥ 0.98** at a non-zero shift → candidate for obs_date/known_at correction.",
        "- **corr ≥ 0.80** → proxy-grade; timing adjustment optional.",
        "- **corr < 0.80** → below proxy threshold; consider replacing the source.",
        "",
        "## Proposed Correction Rules",
        "",
        "*(Populate after running `mad-data timing-diagnostic` against the live database.)*",
        "",
        "No correction rules applied yet — instrument YAML obs_date fields are unchanged.",
        "",
        "## Raw Data Note",
        "",
        "This report is generated from the local fixture database only; live data was not",
        "fetched during generation.  Re-run after `mad-data backfill` to refresh.",
    ]

    return lines
