"""Coverage and fidelity report for all instruments.

Reads price data from the database only (no live provider calls).
Compares each instrument's chosen live source against its bloomberg_ews
reference series over the overlap period.

Metrics
-------
- Weekly-change correlation: log returns for price/index_points, differences
  for percent (yields).  Computed over aligned EWS Tuesdays (or monthly
  periods for monthly-frequency live sources).
- Median level ratio: live / ews, only for exact-fidelity live sources.
  Catches unit and scale errors that leave correlation unchanged.
- Sign check: correlation < -0.5 → likely inverted quote direction.
- Overlap: number of aligned weeks (or months for monthly sources).

Thresholds
----------
- overlap < 104 equivalent-weeks → "insufficient"; no suggestion
- correlation ≥ 0.98 → suggest "exact"
- correlation ≥ 0.80 → suggest "proxy"
- correlation < 0.80 → suggest "review_required"
- correlation < -0.50 → also set sign_inverted = True

--check exit-code rule
----------------------
Exit 1 if any instrument's YAML fidelity differs from the suggestion, unless
the instrument's ``notes`` field contains the substring "fidelity-override:".
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .ingestion import is_stale

log = logging.getLogger(__name__)

_GAP_BDAYS: int = 5
_MIN_OVERLAP_WEEKS: int = 104
_MONTHLY_MIN_OVERLAP: int = 24  # months; 24 * 4 = 96 ≈ 104 weeks
_EXACT_THRESHOLD: float = 0.98
_PROXY_THRESHOLD: float = 0.80
_INVERSION_THRESHOLD: float = -0.50


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class GapRecord:
    start: date
    end: date
    bdays: int


@dataclass
class FidelityComparison:
    overlap_periods: int  # weeks for daily/weekly sources, months for monthly
    period_unit: str  # "weeks" or "months"
    overlap_weeks_approx: int  # for the 104-week threshold; months * 4 for monthly
    sufficient: bool
    change_correlation: Optional[float]
    median_level_ratio: Optional[float]  # None when is_exact=False or insufficient
    sign_inverted: bool
    suggested_fidelity: str  # "exact" | "proxy" | "review_required" | "insufficient"


@dataclass
class CoverageRow:
    instrument_id: str
    instrument_name: str
    asset_class: str
    units: str
    # Live source from YAML + DB
    chosen_provider: Optional[str]
    chosen_symbol: Optional[str]
    yaml_fidelity: Optional[str]
    # Coverage stats
    first_date: Optional[date]
    last_date: Optional[date]
    row_count: int
    gaps: list[GapRecord] = field(default_factory=list)
    is_stale_flag: bool = True
    # Comparison against bloomberg_ews (None when no EWS or no live data)
    comparison: Optional[FidelityComparison] = None
    # --check fields
    fidelity_mismatch: bool = False
    has_fidelity_override: bool = False


@dataclass
class CoverageReport:
    rows: list[CoverageRow]
    generated_at: date


# ── Public helpers (testable offline, no DB) ──────────────────────────────────


def find_gaps(obs_dates: list[date], min_bdays: int = _GAP_BDAYS) -> list[GapRecord]:
    """Return gaps longer than *min_bdays* business days in an obs_date list."""
    sorted_dates = sorted(obs_dates)
    gaps: list[GapRecord] = []
    for i in range(1, len(sorted_dates)):
        a = sorted_dates[i - 1]
        b = sorted_dates[i]
        bdays = int(np.busday_count(a.isoformat(), b.isoformat()))
        if bdays > min_bdays:
            gaps.append(GapRecord(start=a, end=b, bdays=bdays))
    return gaps


def compare_fidelity(
    live_df: pd.DataFrame,
    ews_df: pd.DataFrame,
    *,
    units: str,
    source_frequency: str,
    is_exact: bool,
) -> FidelityComparison:
    """Compare a live series against the bloomberg_ews reference.

    Parameters
    ----------
    live_df:
        DataFrame with columns ``obs_date`` (date) and ``value`` (float).
        May be at daily or monthly frequency depending on *source_frequency*.
    ews_df:
        Same structure; bloomberg_ews series (weekly Tuesdays).
    units:
        Instrument's units: ``"price"``, ``"index_points"``, or ``"percent"``.
    source_frequency:
        Live source's native frequency: ``"daily"``, ``"weekly"``, or
        ``"monthly"``.  Monthly sources are compared at monthly granularity;
        all others are aligned to EWS Tuesdays with a backward-fill join.
    is_exact:
        True when the YAML fidelity for this live source is ``"exact"``.
        Controls whether the level ratio is computed.
    """
    if source_frequency == "monthly":
        aligned = _align_monthly(live_df, ews_df)
        period_unit = "months"
        overlap_periods = len(aligned)
        overlap_weeks_approx = overlap_periods * 4
        sufficient = overlap_periods >= _MONTHLY_MIN_OVERLAP
    else:
        aligned = _align_weekly(live_df, ews_df)
        period_unit = "weeks"
        overlap_periods = len(aligned)
        overlap_weeks_approx = overlap_periods
        sufficient = overlap_periods >= _MIN_OVERLAP_WEEKS

    if not sufficient or len(aligned) < 4:
        return FidelityComparison(
            overlap_periods=overlap_periods,
            period_unit=period_unit,
            overlap_weeks_approx=overlap_weeks_approx,
            sufficient=False,
            change_correlation=None,
            median_level_ratio=None,
            sign_inverted=False,
            suggested_fidelity="insufficient",
        )

    live_vals = aligned["live_value"].reset_index(drop=True)
    ews_vals = aligned["ews_value"].reset_index(drop=True)

    live_chg = _compute_changes(live_vals, units)
    ews_chg = _compute_changes(ews_vals, units)
    corr = _correlation(live_chg, ews_chg)

    sign_inverted = corr is not None and corr < _INVERSION_THRESHOLD
    level_ratio = _level_ratio(live_vals, ews_vals) if is_exact else None

    if corr is None:
        suggested = "review_required"
    elif corr >= _EXACT_THRESHOLD:
        suggested = "exact"
    elif corr >= _PROXY_THRESHOLD:
        suggested = "proxy"
    else:
        suggested = "review_required"

    return FidelityComparison(
        overlap_periods=overlap_periods,
        period_unit=period_unit,
        overlap_weeks_approx=overlap_weeks_approx,
        sufficient=True,
        change_correlation=corr,
        median_level_ratio=level_ratio,
        sign_inverted=sign_inverted,
        suggested_fidelity=suggested,
    )


# ── Report builder (requires DB session) ─────────────────────────────────────


def build_report(
    session,
    instruments,
    today: Optional[date] = None,
) -> CoverageReport:
    """Build a coverage report from the database.

    Parameters
    ----------
    session:
        SQLAlchemy ``Session`` with data already loaded (after backfill/load-ews).
    instruments:
        List of ``Instrument`` objects from ``load_registry()``.
    today:
        Reference date for staleness; defaults to ``date.today()``.
    """
    if today is None:
        today = date.today()

    prices = _load_prices(session)

    rows: list[CoverageRow] = []
    for inst in instruments:
        rows.append(_build_row(inst, prices, today))

    # Unoverridden mismatches first, then alphabetical by ID
    rows.sort(
        key=lambda r: (not (r.fidelity_mismatch and not r.has_fidelity_override), r.instrument_id)
    )
    return CoverageReport(rows=rows, generated_at=today)


# ── Writers ───────────────────────────────────────────────────────────────────


def write_markdown(report: CoverageReport, path: Path) -> None:
    """Write the markdown coverage report to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_render_markdown(report)) + "\n", encoding="utf-8")


def write_csv(report: CoverageReport, path: Path) -> None:
    """Write a flat CSV coverage report to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instrument_id",
        "instrument_name",
        "asset_class",
        "units",
        "chosen_provider",
        "chosen_symbol",
        "yaml_fidelity",
        "first_date",
        "last_date",
        "row_count",
        "gap_count",
        "is_stale",
        "overlap_periods",
        "period_unit",
        "overlap_weeks_approx",
        "sufficient_overlap",
        "change_correlation",
        "median_level_ratio",
        "sign_inverted",
        "suggested_fidelity",
        "fidelity_mismatch",
        "has_fidelity_override",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in report.rows:
            cmp = r.comparison
            writer.writerow(
                {
                    "instrument_id": r.instrument_id,
                    "instrument_name": r.instrument_name,
                    "asset_class": r.asset_class,
                    "units": r.units,
                    "chosen_provider": r.chosen_provider or "",
                    "chosen_symbol": r.chosen_symbol or "",
                    "yaml_fidelity": r.yaml_fidelity or "",
                    "first_date": str(r.first_date) if r.first_date else "",
                    "last_date": str(r.last_date) if r.last_date else "",
                    "row_count": r.row_count,
                    "gap_count": len(r.gaps),
                    "is_stale": r.is_stale_flag,
                    "overlap_periods": cmp.overlap_periods if cmp else "",
                    "period_unit": cmp.period_unit if cmp else "",
                    "overlap_weeks_approx": cmp.overlap_weeks_approx if cmp else "",
                    "sufficient_overlap": cmp.sufficient if cmp else "",
                    "change_correlation": (
                        f"{cmp.change_correlation:.4f}"
                        if cmp and cmp.change_correlation is not None
                        else ""
                    ),
                    "median_level_ratio": (
                        f"{cmp.median_level_ratio:.4f}"
                        if cmp and cmp.median_level_ratio is not None
                        else ""
                    ),
                    "sign_inverted": cmp.sign_inverted if cmp else "",
                    "suggested_fidelity": cmp.suggested_fidelity if cmp else "",
                    "fidelity_mismatch": r.fidelity_mismatch,
                    "has_fidelity_override": r.has_fidelity_override,
                }
            )


# ── Internal: DB query ────────────────────────────────────────────────────────


def _load_prices(session) -> dict[tuple[str, str], pd.DataFrame]:
    """Load all price rows grouped by (instrument_id, provider)."""
    from sqlalchemy import select

    from .db.models import PriceRow

    rows = session.execute(
        select(
            PriceRow.instrument_id,
            PriceRow.provider,
            PriceRow.obs_date,
            PriceRow.value,
        ).order_by(PriceRow.instrument_id, PriceRow.provider, PriceRow.obs_date)
    ).fetchall()

    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r.instrument_id, r.provider)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append({"obs_date": r.obs_date, "value": float(r.value)})

    return {k: pd.DataFrame(v) for k, v in grouped.items()}


# ── Internal: row builder ─────────────────────────────────────────────────────


def _build_row(
    inst,
    prices: dict[tuple[str, str], pd.DataFrame],
    today: date,
) -> CoverageRow:
    """Build one CoverageRow for *inst* from the pre-loaded *prices* dict."""
    # ── Chosen live source: lowest priority non-bloomberg_ews with DB data ────
    live_sources = sorted(
        [s for s in inst.sources if s.provider != "bloomberg_ews" and s.symbol],
        key=lambda s: s.priority,
    )
    chosen_src = None
    chosen_df: Optional[pd.DataFrame] = None
    for src in live_sources:
        df = prices.get((inst.id, src.provider))
        if df is not None and not df.empty:
            chosen_src = src
            chosen_df = df
            break

    # ── Coverage stats ────────────────────────────────────────────────────────
    if chosen_df is not None and not chosen_df.empty:
        obs_dates: list[date] = chosen_df["obs_date"].tolist()
        first_date: Optional[date] = min(obs_dates)
        last_date: Optional[date] = max(obs_dates)
        row_count = len(chosen_df)
        gaps = find_gaps(obs_dates)
        freq = (
            chosen_src.source_frequency.value
            if chosen_src is not None and chosen_src.source_frequency is not None
            else inst.native_frequency.value
        )
        stale = is_stale(last_date, freq, today)
    else:
        first_date = last_date = None
        row_count = 0
        gaps = []
        stale = True

    # ── bloomberg_ews comparison ──────────────────────────────────────────────
    ews_df = prices.get((inst.id, "bloomberg_ews"))
    comparison: Optional[FidelityComparison] = None

    if chosen_src is not None and chosen_df is not None and ews_df is not None and not ews_df.empty:
        src_freq = (
            chosen_src.source_frequency.value
            if chosen_src.source_frequency is not None
            else "daily"
        )
        is_exact = chosen_src.fidelity.value == "exact"
        comparison = compare_fidelity(
            chosen_df,
            ews_df,
            units=inst.units.value,
            source_frequency=src_freq,
            is_exact=is_exact,
        )

    # ── Fidelity mismatch ─────────────────────────────────────────────────────
    yaml_fidelity = chosen_src.fidelity.value if chosen_src is not None else None
    has_override = "fidelity-override:" in inst.notes

    mismatch = False
    if (
        comparison is not None
        and comparison.sufficient
        and yaml_fidelity is not None
        and comparison.suggested_fidelity != "insufficient"
    ):
        mismatch = yaml_fidelity != comparison.suggested_fidelity

    return CoverageRow(
        instrument_id=inst.id,
        instrument_name=inst.name,
        asset_class=inst.asset_class.value,
        units=inst.units.value,
        chosen_provider=chosen_src.provider if chosen_src else None,
        chosen_symbol=chosen_src.symbol if chosen_src else None,
        yaml_fidelity=yaml_fidelity,
        first_date=first_date,
        last_date=last_date,
        row_count=row_count,
        gaps=gaps,
        is_stale_flag=stale,
        comparison=comparison,
        fidelity_mismatch=mismatch,
        has_fidelity_override=has_override,
    )


# ── Internal: alignment ───────────────────────────────────────────────────────


def _align_weekly(live_df: pd.DataFrame, ews_df: pd.DataFrame) -> pd.DataFrame:
    """Align live series to EWS Tuesdays via backward-fill (last obs ≤ Tuesday).

    merge_asof requires a numeric or datetime key, so obs_date (Python date)
    is converted to Timestamp before merging.
    """

    def _to_dt(df: pd.DataFrame, val_col: str) -> pd.DataFrame:
        out = df[["obs_date", "value"]].copy()
        out["_dt"] = pd.to_datetime(out["obs_date"])
        return out.sort_values("_dt").rename(columns={"value": val_col})

    ews_s = _to_dt(ews_df, "ews_value")
    live_s = _to_dt(live_df, "live_value")

    merged = pd.merge_asof(
        ews_s[["_dt", "ews_value"]],
        live_s[["_dt", "live_value"]],
        on="_dt",
        direction="backward",
    )
    return merged.dropna(subset=["ews_value", "live_value"]).reset_index(drop=True)


def _align_monthly(live_df: pd.DataFrame, ews_df: pd.DataFrame) -> pd.DataFrame:
    """Resample EWS weekly → monthly mean; align with monthly live series."""
    ews_monthly = (
        ews_df.assign(_dt=pd.to_datetime(ews_df["obs_date"]))
        .set_index("_dt")["value"]
        .resample("MS")
        .mean()
        .dropna()
        .reset_index()
        .rename(columns={"_dt": "_dt_ews", "value": "ews_value"})
    )
    ews_monthly["_ym"] = ews_monthly["_dt_ews"].dt.to_period("M")

    live_m = live_df.copy().rename(columns={"value": "live_value"})
    live_m["_ym"] = pd.to_datetime(live_m["obs_date"]).dt.to_period("M")

    merged = pd.merge(
        ews_monthly[["_ym", "ews_value"]],
        live_m[["_ym", "live_value"]],
        on="_ym",
    )
    return merged.dropna(subset=["ews_value", "live_value"]).reset_index(drop=True)


# ── Internal: metric helpers ──────────────────────────────────────────────────


def _compute_changes(series: pd.Series, units: str) -> pd.Series:
    """Log returns for price/index_points; first differences for percent."""
    if units in ("price", "index_points"):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(series.where(series > 0)).diff()
    return series.diff()  # percent / yields


def _correlation(a: pd.Series, b: pd.Series) -> Optional[float]:
    """Pearson correlation dropping NaN pairs; None when < 4 valid pairs."""
    combined = pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1).dropna()
    combined.columns = pd.Index(["a", "b"])
    if len(combined) < 4:
        return None
    corr = combined["a"].corr(combined["b"])
    return float(corr) if pd.notna(corr) else None


def _level_ratio(live: pd.Series, ews: pd.Series) -> Optional[float]:
    """Median of live / ews, skipping zero-denominator rows."""
    denom = ews.where(ews != 0, other=np.nan)
    valid = (live / denom).dropna()
    return float(valid.median()) if not valid.empty else None


# ── Internal: markdown renderer ───────────────────────────────────────────────


def _render_markdown(report: CoverageReport) -> list[str]:
    lines: list[str] = []
    lines += [
        "# Symbol Coverage Report",
        "",
        f"*Generated {report.generated_at} from the database only.*",
        "",
    ]

    n_live = sum(1 for r in report.rows if r.chosen_provider)
    n_unavail = len(report.rows) - n_live
    mismatches = [r for r in report.rows if r.fidelity_mismatch and not r.has_fidelity_override]
    n_stale = sum(1 for r in report.rows if r.is_stale_flag)

    lines += [
        "## Summary",
        "",
        f"- {len(report.rows)} instruments total",
        f"- {n_live} with live data, {n_unavail} unavailable",
        f"- {len(mismatches)} fidelity mismatch(es) requiring review",
        f"- {n_stale} stale instrument(s)",
        "",
    ]

    if mismatches:
        lines += [
            "## Fidelity Mismatches (review required)",
            "",
            "| Instrument | Provider | Symbol | YAML | Suggested | Corr | Level Ratio | Inverted |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in mismatches:
            cmp = r.comparison
            corr_s = (
                f"{cmp.change_correlation:.3f}"
                if cmp and cmp.change_correlation is not None
                else "—"
            )
            ratio_s = (
                f"{cmp.median_level_ratio:.3f}"
                if cmp and cmp.median_level_ratio is not None
                else "—"
            )
            inv_s = "yes" if cmp and cmp.sign_inverted else "no"
            sugg_s = cmp.suggested_fidelity if cmp else "—"
            lines.append(
                f"| {r.instrument_id} | {r.chosen_provider or '—'} | {r.chosen_symbol or '—'}"
                f" | {r.yaml_fidelity or '—'} | {sugg_s} | {corr_s} | {ratio_s} | {inv_s} |"
            )
        lines.append("")

    lines += [
        "## All Instruments",
        "",
        "| Instrument | Provider | Symbol | YAML | First | Last | Rows | Gaps | Stale"
        " | Overlap | Corr | Level Ratio | Inverted | Suggested |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.rows:
        cmp = r.comparison
        prov = r.chosen_provider or "unavailable"
        sym = r.chosen_symbol or "—"
        fid = r.yaml_fidelity or "—"
        first = str(r.first_date) if r.first_date else "—"
        last = str(r.last_date) if r.last_date else "—"
        rows_s = str(r.row_count)
        gaps_s = str(len(r.gaps))
        stale_s = "yes" if r.is_stale_flag else "no"

        if cmp:
            overlap_s = f"{cmp.overlap_periods} {cmp.period_unit}"
            if not cmp.sufficient:
                overlap_s += " (insufficient)"
            corr_s = f"{cmp.change_correlation:.3f}" if cmp.change_correlation is not None else "—"
            ratio_s = f"{cmp.median_level_ratio:.3f}" if cmp.median_level_ratio is not None else "—"
            inv_s = "yes" if cmp.sign_inverted else "no"
            sugg_s = cmp.suggested_fidelity
        else:
            overlap_s = corr_s = ratio_s = inv_s = sugg_s = "—"

        lines.append(
            f"| {r.instrument_id} | {prov} | {sym} | {fid} | {first} | {last}"
            f" | {rows_s} | {gaps_s} | {stale_s} | {overlap_s} | {corr_s}"
            f" | {ratio_s} | {inv_s} | {sugg_s} |"
        )

    lines += ["", "### Gap Detail", ""]
    any_gaps = any(r.gaps for r in report.rows)
    if any_gaps:
        lines += [
            "| Instrument | Gap Start | Gap End | Business Days |",
            "|---|---|---|---|",
        ]
        for r in report.rows:
            for g in r.gaps:
                lines.append(f"| {r.instrument_id} | {g.start} | {g.end} | {g.bdays} |")
    else:
        lines.append("No gaps over 5 business days found.")

    return lines
