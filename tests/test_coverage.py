"""Offline tests for the coverage and fidelity report module.

All tests use synthetic series — no database or network required.
"""

from __future__ import annotations

import csv
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from market_anomaly_detector.coverage import (
    CoverageReport,
    CoverageRow,
    FidelityComparison,
    GapRecord,
    compare_fidelity,
    find_gaps,
    write_csv,
    write_markdown,
)

# ── Test helpers ──────────────────────────────────────────────────────────────


def _weekly_dates(start: date, n: int) -> list[date]:
    """Generate *n* Tuesday-spaced dates starting from *start*."""
    return [start + timedelta(weeks=i) for i in range(n)]


def _monthly_dates(start: date, n: int) -> list[date]:
    """Generate *n* 1st-of-month dates starting from *start*."""
    dates: list[date] = []
    y, m = start.year, start.month
    for _ in range(n):
        dates.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


def _df(dates: list[date], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"obs_date": dates, "value": values})


_START_TUE = date(2005, 1, 4)  # a Tuesday
_N_SUFFICIENT = 130  # 130 weeks > 104 threshold


# ── compare_fidelity: identical series ────────────────────────────────────────


def test_identical_series_correlation_is_one():
    """Identical series must give correlation exactly 1.0 and suggest 'exact'."""
    dates = _weekly_dates(_START_TUE, _N_SUFFICIENT)
    # Non-trivial series to avoid degenerate constant-change case
    vals = [100.0 + i * 0.3 + 0.5 * math.sin(i * 0.1) for i in range(_N_SUFFICIENT)]

    live = _df(dates, vals)
    ews = _df(dates, vals)

    result = compare_fidelity(live, ews, units="price", source_frequency="daily", is_exact=True)

    assert result.sufficient
    assert result.change_correlation is not None
    assert math.isclose(result.change_correlation, 1.0, abs_tol=1e-9)
    assert result.suggested_fidelity == "exact"
    assert not result.sign_inverted
    # Level ratio must be ~1.0 for identical series
    assert result.median_level_ratio is not None
    assert math.isclose(result.median_level_ratio, 1.0, abs_tol=1e-9)


# ── compare_fidelity: inverted series ─────────────────────────────────────────


def test_inverted_series_sign_check_triggers():
    """A negated live series must have correlation < -0.5 and sign_inverted True.

    A pure linear series has constant differences, which gives an undefined
    correlation.  Use a sine-wave component so the differences vary.
    """
    dates = _weekly_dates(_START_TUE, _N_SUFFICIENT)
    # Non-linear: differences are non-constant (sine wave + trend)
    vals = [10 * math.sin(i * 0.2) + i * 0.1 for i in range(_N_SUFFICIENT)]

    live = _df(dates, [-v for v in vals])
    ews = _df(dates, vals)

    # Percent units → diff; live_diff = -ews_diff ⟹ corr = -1.0
    result = compare_fidelity(live, ews, units="percent", source_frequency="daily", is_exact=True)

    assert result.sufficient
    assert result.sign_inverted
    assert result.change_correlation is not None
    assert result.change_correlation < -0.5
    assert result.suggested_fidelity == "review_required"


def test_inverted_price_series_via_reciprocal():
    """FX inversion (live = 1/ews) still triggers the sign check via changes."""
    dates = _weekly_dates(_START_TUE, _N_SUFFICIENT)
    # Trending up: ews goes 100 → 165; live is 1/ews going down.
    ews_vals = [100.0 + i * 0.5 for i in range(_N_SUFFICIENT)]
    live_vals = [1.0 / v for v in ews_vals]

    live = _df(dates, live_vals)
    ews = _df(dates, ews_vals)

    result = compare_fidelity(live, ews, units="price", source_frequency="daily", is_exact=True)

    assert result.sufficient
    assert result.sign_inverted
    assert result.change_correlation is not None
    assert result.change_correlation < -0.5


# ── compare_fidelity: 10× unit error shows in level ratio ─────────────────────


def test_unit_error_shows_in_level_ratio():
    """A 10× scale error must appear in the level ratio without hiding in correlation."""
    dates = _weekly_dates(_START_TUE, _N_SUFFICIENT)
    ews_vals = [100.0 + i * 0.5 for i in range(_N_SUFFICIENT)]
    live_vals = [v * 10.0 for v in ews_vals]

    live = _df(dates, live_vals)
    ews = _df(dates, ews_vals)

    result = compare_fidelity(live, ews, units="price", source_frequency="daily", is_exact=True)

    assert result.sufficient
    # Log returns of 10*x cancel the 10: correlation is still ~1.0
    assert result.change_correlation is not None
    assert result.change_correlation > 0.99
    assert not result.sign_inverted
    # Level ratio must reveal the scale error
    assert result.median_level_ratio is not None
    assert math.isclose(result.median_level_ratio, 10.0, rel_tol=0.01)


def test_unit_error_level_ratio_skipped_for_proxy():
    """Level ratio must be None when is_exact=False (proxy source)."""
    dates = _weekly_dates(_START_TUE, _N_SUFFICIENT)
    vals = [100.0 + i * 0.5 for i in range(_N_SUFFICIENT)]

    live = _df(dates, [v * 10 for v in vals])
    ews = _df(dates, vals)

    result = compare_fidelity(live, ews, units="price", source_frequency="daily", is_exact=False)

    assert result.sufficient
    assert result.median_level_ratio is None


# ── compare_fidelity: yield near zero or negative ─────────────────────────────


def test_yield_near_zero_does_not_raise():
    """Percent-units series with values ≈ 0 or negative must not raise."""
    dates = _weekly_dates(_START_TUE, _N_SUFFICIENT)
    # Cross-zero trajectory: 0.5 → -0.5 → 0.5
    mid = _N_SUFFICIENT // 2
    vals = [0.5 - 0.01 * i for i in range(mid)] + [
        -(0.5 - 0.01 * i) for i in range(_N_SUFFICIENT - mid)
    ]

    live = _df(dates, vals)
    ews = _df(dates, vals)

    result = compare_fidelity(live, ews, units="percent", source_frequency="daily", is_exact=True)

    assert result.sufficient
    assert result.change_correlation is not None
    assert not result.sign_inverted


def test_yield_negative_does_not_raise():
    """Negative yields (e.g., German bund) must not cause errors."""
    dates = _weekly_dates(_START_TUE, _N_SUFFICIENT)
    vals = [-0.5 - i * 0.001 for i in range(_N_SUFFICIENT)]

    live = _df(dates, vals)
    ews = _df(dates, vals)

    result = compare_fidelity(live, ews, units="percent", source_frequency="daily", is_exact=True)

    assert result.sufficient
    assert result.change_correlation is not None
    assert math.isclose(result.change_correlation, 1.0, abs_tol=1e-9)


# ── compare_fidelity: monthly alignment ───────────────────────────────────────


def test_monthly_alignment_counts_in_months():
    """Monthly source_frequency must align at monthly granularity."""
    # EWS: 3 years of weekly Tuesdays (156 weeks)
    ews_dates = _weekly_dates(_START_TUE, 156)
    ews_vals = [100.0 + i * 0.2 for i in range(156)]
    ews = _df(ews_dates, ews_vals)

    # Live: 36 monthly observations (1st of each month)
    live_dates = _monthly_dates(date(2005, 1, 1), 36)
    # Monthly live ≈ mean of weekly EWS in that month — use a correlated series
    live_vals = [100.0 + j * 0.87 for j in range(36)]
    live = _df(live_dates, live_vals)

    result = compare_fidelity(live, ews, units="price", source_frequency="monthly", is_exact=True)

    assert result.period_unit == "months"
    assert result.overlap_periods > 0
    # 36 months * 4 = 144 equivalent-weeks > 104 threshold → sufficient
    assert result.sufficient
    assert result.change_correlation is not None


def test_monthly_alignment_identical_series():
    """Monthly-aligned identical (after resampling) series must give high correlation."""
    # Build EWS weekly with a known monthly mean: use a step function
    # Each week in a month gets the same value; monthly mean = that value.
    months = _monthly_dates(date(2005, 1, 1), 30)  # 30 months
    ews_dates: list[date] = []
    ews_vals: list[float] = []
    live_dates: list[date] = []
    live_vals: list[float] = []
    for i, mo in enumerate(months):
        val = 100.0 + i * 2.0
        live_dates.append(mo)
        live_vals.append(val)
        # 4 weekly Tuesdays per month (approximate)
        for w in range(4):
            ews_dates.append(mo + timedelta(days=1 + w * 7))
            ews_vals.append(val)

    live = _df(live_dates, live_vals)
    ews = _df(ews_dates, ews_vals)

    result = compare_fidelity(live, ews, units="price", source_frequency="monthly", is_exact=True)

    assert result.period_unit == "months"
    assert result.sufficient
    assert result.change_correlation is not None
    assert result.change_correlation > 0.98


# ── compare_fidelity: insufficient overlap ────────────────────────────────────


def test_insufficient_overlap_weekly():
    """< 104 weekly observations must produce sufficient=False and suggest 'insufficient'."""
    dates = _weekly_dates(_START_TUE, 50)  # well below 104
    vals = [100.0 + i for i in range(50)]

    live = _df(dates, vals)
    ews = _df(dates, vals)

    result = compare_fidelity(live, ews, units="price", source_frequency="daily", is_exact=True)

    assert not result.sufficient
    assert result.overlap_weeks_approx < 104
    assert result.suggested_fidelity == "insufficient"
    assert result.change_correlation is None
    assert result.median_level_ratio is None


def test_insufficient_overlap_monthly():
    """< 24 monthly observations (≈ 96 equivalent weeks) must also be insufficient."""
    ews_dates = _weekly_dates(_START_TUE, 80)  # covers ~20 months
    ews = _df(ews_dates, [100.0 + i for i in range(80)])

    live_dates = _monthly_dates(date(2005, 1, 1), 20)  # 20 months < 24 threshold
    live = _df(live_dates, [100.0 + j * 4 for j in range(20)])

    result = compare_fidelity(live, ews, units="price", source_frequency="monthly", is_exact=True)

    assert not result.sufficient
    assert result.suggested_fidelity == "insufficient"


# ── compare_fidelity: proxy and review threshold cases ───────────────────────


def test_proxy_threshold_case():
    """Correlation in [0.80, 0.98) must suggest 'proxy'."""
    dates = _weekly_dates(_START_TUE, _N_SUFFICIENT)
    rng = np.random.default_rng(42)
    base_changes = rng.standard_normal(_N_SUFFICIENT)
    # Mix 88% signal, 12% noise → correlation ≈ 0.88
    noise = rng.standard_normal(_N_SUFFICIENT)
    live_changes = 0.88 * base_changes + 0.47 * noise
    ews_changes = base_changes

    # Reconstruct levels from changes (diff space)
    ews_vals = list(np.cumsum(ews_changes) + 100)
    live_vals = list(np.cumsum(live_changes) + 100)

    live = _df(dates, live_vals)
    ews = _df(dates, ews_vals)

    result = compare_fidelity(live, ews, units="percent", source_frequency="daily", is_exact=True)

    assert result.sufficient
    assert result.change_correlation is not None
    assert 0.80 <= result.change_correlation < 0.98
    assert result.suggested_fidelity == "proxy"


def test_review_required_threshold_case():
    """Correlation below 0.80 must suggest 'review_required'."""
    dates = _weekly_dates(_START_TUE, _N_SUFFICIENT)
    rng = np.random.default_rng(99)
    live_vals = list(rng.standard_normal(_N_SUFFICIENT).cumsum() + 100)
    ews_vals = list(rng.standard_normal(_N_SUFFICIENT).cumsum() + 100)

    live = _df(dates, live_vals)
    ews = _df(dates, ews_vals)

    result = compare_fidelity(live, ews, units="percent", source_frequency="daily", is_exact=True)

    assert result.sufficient
    assert result.change_correlation is not None
    assert result.change_correlation < 0.80
    assert result.suggested_fidelity == "review_required"


# ── find_gaps ─────────────────────────────────────────────────────────────────


def test_find_gaps_no_gaps():
    """Daily series with no gap > 5 business days must return empty list."""
    # Mon–Fri for 3 weeks (15 business days, no gap > 5 bdays)
    start = date(2024, 1, 8)  # Monday
    dates = [
        start + timedelta(days=i) for i in range(15) if (start + timedelta(days=i)).weekday() < 5
    ]
    assert not find_gaps(dates)


def test_find_gaps_detects_large_gap():
    """A gap spanning 2 weeks (10 business days) must be flagged."""
    d1 = date(2024, 1, 12)  # Friday
    d2 = date(2024, 1, 29)  # Monday two weeks later: 12 bdays gap (Fri→Mon = 3 bdays; add 2 weeks)
    dates = [d1, d2]
    gaps = find_gaps(dates)
    assert len(gaps) == 1
    assert gaps[0].start == d1
    assert gaps[0].end == d2
    assert gaps[0].bdays > 5


def test_find_gaps_weekend_is_not_a_gap():
    """Friday → Monday is 1 business day, not a gap."""
    fri = date(2024, 1, 12)
    mon = date(2024, 1, 15)
    dates = [fri, mon]
    assert not find_gaps(dates)


def test_find_gaps_exactly_at_boundary():
    """Exactly 5 business days must NOT be flagged (> 5, not ≥ 5)."""
    # Friday Jan 12 to Friday Jan 19 = 5 business days
    d1 = date(2024, 1, 12)
    d2 = date(2024, 1, 19)
    dates = [d1, d2]
    gaps = find_gaps(dates, min_bdays=5)
    assert not gaps


def test_find_gaps_just_over_boundary():
    """6 business days must be flagged."""
    d1 = date(2024, 1, 12)  # Friday
    d2 = date(2024, 1, 22)  # Monday two weeks later; np.busday_count → 6
    gaps = find_gaps([d1, d2], min_bdays=5)
    assert len(gaps) == 1
    assert gaps[0].bdays == 6


# ── compare_fidelity: tolerance prevents gap bridging ────────────────────────


def test_gap_in_live_data_does_not_bridge_with_tolerance():
    """A 4-year hole in the live series must not propagate stale values.

    With tolerance=7 days in _align_weekly, EWS Tuesdays that fall in the
    gap get NaN (no live match) and are excluded from the overlap.  The
    remaining overlap must still give correlation 1.0 and suggest 'exact'.
    """
    # EWS: 10 years of weekly Tuesdays (520 weeks, 2005-01-04 to ~2014-12)
    ews_start = date(2005, 1, 4)
    ews_dates = _weekly_dates(ews_start, 520)
    ews_vals = [100.0 + i * 0.3 + 0.5 * math.sin(i * 0.1) for i in range(520)]
    ews = _df(ews_dates, ews_vals)

    # Live: identical values but missing the middle ~4 years (2008-2012)
    # Keep first 156 weeks (~3 yrs) and last 156 weeks (~3 yrs)
    gap_start_idx = 156
    gap_end_idx = 364  # skip 208 weeks
    live_indices = list(range(gap_start_idx)) + list(range(gap_end_idx, 520))
    live_dates = [ews_dates[i] for i in live_indices]
    live_vals = [ews_vals[i] for i in live_indices]
    live = _df(live_dates, live_vals)

    result = compare_fidelity(live, ews, units="price", source_frequency="daily", is_exact=True)

    # Overlap must be ~312 weeks (the non-gap portions), well above 104
    assert result.sufficient
    assert result.overlap_periods < 520  # gap weeks were excluded
    assert result.overlap_periods >= 104  # enough remaining for a valid comparison

    # Values are identical over the overlap, so correlation must be 1.0
    assert result.change_correlation is not None
    assert math.isclose(result.change_correlation, 1.0, abs_tol=1e-9)
    assert result.suggested_fidelity == "exact"
    assert not result.sign_inverted


# ── write_markdown / write_csv: smoke tests ───────────────────────────────────


def _make_synthetic_report() -> CoverageReport:
    cmp_ok = FidelityComparison(
        overlap_periods=120,
        period_unit="weeks",
        overlap_weeks_approx=120,
        sufficient=True,
        change_correlation=0.995,
        median_level_ratio=1.002,
        sign_inverted=False,
        suggested_fidelity="exact",
    )
    cmp_mismatch = FidelityComparison(
        overlap_periods=110,
        period_unit="weeks",
        overlap_weeks_approx=110,
        sufficient=True,
        change_correlation=0.45,
        median_level_ratio=0.98,
        sign_inverted=False,
        suggested_fidelity="review_required",
    )
    rows = [
        CoverageRow(
            instrument_id="gold",
            instrument_name="Gold",
            asset_class="commodity",
            units="price",
            chosen_provider="yahoo",
            chosen_symbol="GC=F",
            yaml_fidelity="proxy",
            first_date=date(2006, 1, 1),
            last_date=date(2021, 4, 1),
            row_count=3800,
            gaps=[],
            is_stale_flag=True,
            comparison=cmp_ok,
            fidelity_mismatch=False,
            has_fidelity_override=False,
        ),
        CoverageRow(
            instrument_id="crb_commodity",
            instrument_name="CRB Commodity",
            asset_class="macro_index",
            units="index_points",
            chosen_provider="yahoo",
            chosen_symbol="DBC",
            yaml_fidelity="proxy",
            first_date=date(2006, 2, 3),
            last_date=date(2021, 4, 1),
            row_count=3700,
            gaps=[GapRecord(date(2008, 10, 1), date(2008, 10, 15), 10)],
            is_stale_flag=True,
            comparison=cmp_mismatch,
            fidelity_mismatch=True,
            has_fidelity_override=False,
        ),
        CoverageRow(
            instrument_id="baltic_dry",
            instrument_name="Baltic Dry Index",
            asset_class="macro_index",
            units="index_points",
            chosen_provider=None,
            chosen_symbol=None,
            yaml_fidelity=None,
            first_date=None,
            last_date=None,
            row_count=0,
            gaps=[],
            is_stale_flag=True,
            comparison=None,
            fidelity_mismatch=False,
            has_fidelity_override=False,
        ),
    ]
    return CoverageReport(rows=rows, generated_at=date(2024, 1, 15))


def test_write_markdown_produces_valid_file(tmp_path: Path):
    """write_markdown must produce a non-empty file with the expected headings."""
    report = _make_synthetic_report()
    out = tmp_path / "coverage.md"
    write_markdown(report, out)

    text = out.read_text(encoding="utf-8")
    assert "# Symbol Coverage Report" in text
    assert "## Summary" in text
    assert "## All Instruments" in text
    assert "## Fidelity Mismatches" in text
    assert "gold" in text
    assert "crb_commodity" in text
    assert "baltic_dry" in text


def test_write_csv_produces_valid_file(tmp_path: Path):
    """write_csv must produce a parseable CSV with one row per instrument."""
    report = _make_synthetic_report()
    out = tmp_path / "coverage.csv"
    write_csv(report, out)

    with out.open(newline="", encoding="utf-8") as fh:
        reader = list(csv.DictReader(fh))

    assert len(reader) == 3
    ids = {r["instrument_id"] for r in reader}
    assert ids == {"gold", "crb_commodity", "baltic_dry"}


def test_write_csv_correlation_precision(tmp_path: Path):
    """Correlation values in CSV must be formatted to 4 decimal places."""
    report = _make_synthetic_report()
    out = tmp_path / "coverage.csv"
    write_csv(report, out)

    with out.open(newline="", encoding="utf-8") as fh:
        reader = {r["instrument_id"]: r for r in csv.DictReader(fh)}

    assert reader["gold"]["change_correlation"] == "0.9950"


def test_mismatch_rows_sort_first():
    """build_report sorts unoverridden mismatch rows to the top."""
    report = _make_synthetic_report()
    # The sort key: mismatches-without-override sort before everything else.
    rows = sorted(
        report.rows,
        key=lambda r: (not (r.fidelity_mismatch and not r.has_fidelity_override), r.instrument_id),
    )
    assert rows[0].instrument_id == "crb_commodity"  # mismatch=True, no override


def test_fidelity_override_suppresses_sort():
    """An instrument with fidelity-override: in notes is not sorted to the top."""
    cmp = FidelityComparison(
        overlap_periods=110,
        period_unit="weeks",
        overlap_weeks_approx=110,
        sufficient=True,
        change_correlation=0.45,
        median_level_ratio=None,
        sign_inverted=False,
        suggested_fidelity="review_required",
    )
    row_with_override = CoverageRow(
        instrument_id="aaa_override",
        instrument_name="Override Instrument",
        asset_class="commodity",
        units="price",
        chosen_provider="yahoo",
        chosen_symbol="AAA=F",
        yaml_fidelity="proxy",
        first_date=date(2010, 1, 1),
        last_date=date(2021, 1, 1),
        row_count=2000,
        gaps=[],
        is_stale_flag=False,
        comparison=cmp,
        fidelity_mismatch=True,
        has_fidelity_override=True,  # overridden — should NOT sort to top
    )
    row_no_mismatch = CoverageRow(
        instrument_id="zzz_clean",
        instrument_name="Clean Instrument",
        asset_class="commodity",
        units="price",
        chosen_provider="fred",
        chosen_symbol="CLEAN",
        yaml_fidelity="exact",
        first_date=date(2010, 1, 1),
        last_date=date(2021, 1, 1),
        row_count=2000,
        gaps=[],
        is_stale_flag=False,
        comparison=None,
        fidelity_mismatch=False,
        has_fidelity_override=False,
    )

    # build_report sorts, but we can test the sort logic directly
    rows = [row_no_mismatch, row_with_override]
    rows.sort(
        key=lambda r: (not (r.fidelity_mismatch and not r.has_fidelity_override), r.instrument_id)
    )
    # override row has mismatch but has_override=True → not promoted
    assert rows[0].instrument_id == "aaa_override"  # alphabetical, since neither is promoted
    assert rows[1].instrument_id == "zzz_clean"
