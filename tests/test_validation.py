"""Offline tests for the validation pipeline."""

from __future__ import annotations

from datetime import date

import pandas as pd

from market_anomaly_detector.registry import (
    AssetClass,
    Fidelity,
    Frequency,
    Instrument,
    SourceEntry,
    Units,
)
from market_anomaly_detector.validation import validate_provider_output


def _inst(
    id_: str = "test_inst",
    asset_class: AssetClass = AssetClass.equity_index,
    units: Units = Units.index_points,
    allow_negative: bool = False,
) -> Instrument:
    return Instrument(
        id=id_,
        name="Test",
        asset_class=asset_class,
        groups=["test"],
        units=units,
        quote="points",
        native_frequency=Frequency.daily,
        sources=[SourceEntry(provider="yahoo", symbol="TEST", priority=1, fidelity=Fidelity.exact)],
        allow_negative=allow_negative,
    )


def _df(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_date": [date.fromisoformat(d) for d, _ in rows],
            "value": [float(v) for _, v in rows],
        }
    )


# ── Non-numeric values ────────────────────────────────────────────────────────


def test_validate_nonnumeric_dropped():
    df = pd.DataFrame({"obs_date": [date(2024, 1, 2), date(2024, 1, 3)], "value": ["bad", 100.0]})
    clean, flags = validate_provider_output(df, _inst())
    assert len(clean) == 1
    assert float(clean.iloc[0]["value"]) == 100.0
    assert not flags


# ── Duplicate obs_dates ───────────────────────────────────────────────────────


def test_validate_exact_duplicate_dropped_silently():
    """Exact duplicate (same value) is deduplicated; no flag emitted."""
    df = _df([("2024-01-02", 100.0), ("2024-01-02", 100.0), ("2024-01-03", 101.0)])
    clean, flags = validate_provider_output(df, _inst())
    assert len(clean) == 2
    assert not [f for f in flags if f.rule == "duplicate_dates_disagree"]


def test_validate_conflicting_duplicate_flagged():
    """Conflicting values for the same date produce a duplicate_dates_disagree flag."""
    df = _df([("2024-01-02", 100.0), ("2024-01-02", 999.0), ("2024-01-03", 101.0)])
    clean, flags = validate_provider_output(df, _inst())
    assert len(clean) == 2  # deduplicated to one row per date
    dup_flags = [f for f in flags if f.rule == "duplicate_dates_disagree"]
    assert len(dup_flags) == 1
    assert dup_flags[0].obs_date == date(2024, 1, 2)


# ── Negative values ───────────────────────────────────────────────────────────


def test_validate_negative_flagged_for_equity_index():
    df = _df([("2024-01-02", 100.0), ("2024-01-03", -5.0)])
    clean, flags = validate_provider_output(df, _inst(asset_class=AssetClass.equity_index))
    neg_flags = [f for f in flags if f.rule == "negative_value"]
    assert len(neg_flags) == 1
    assert neg_flags[0].obs_date == date(2024, 1, 3)
    assert len(clean) == 2  # row is still present


def test_validate_negative_allowed_for_rate():
    """Negative rates (yields) should not be flagged."""
    df = _df([("2024-01-02", 5.0), ("2024-01-03", -0.5)])
    clean, flags = validate_provider_output(
        df, _inst(asset_class=AssetClass.rate, units=Units.percent)
    )
    assert not [f for f in flags if f.rule == "negative_value"]
    assert len(clean) == 2


def test_validate_negative_allowed_for_wti_front():
    """WTI crude can go negative at roll; must not be flagged."""
    inst = Instrument(
        id="wti_front",
        name="WTI Crude",
        asset_class=AssetClass.commodity,
        groups=["test"],
        units=Units.price,
        quote="USD/bbl",
        native_frequency=Frequency.daily,
        sources=[SourceEntry(provider="yahoo", symbol="CL=F", priority=1, fidelity=Fidelity.exact)],
        allow_negative=True,
    )
    df = _df([("2020-04-20", -37.63), ("2020-04-21", 10.0)])
    clean, flags = validate_provider_output(df, inst)
    assert not [f for f in flags if f.rule == "negative_value"]
    assert len(clean) == 2


# ── Large single-day moves ────────────────────────────────────────────────────


def test_validate_large_move_flagged():
    """A 20% single-day move on an equity_index (threshold=15%) is flagged."""
    df = _df([("2024-01-02", 100.0), ("2024-01-03", 120.0)])
    clean, flags = validate_provider_output(df, _inst(asset_class=AssetClass.equity_index))
    move_flags = [f for f in flags if f.rule == "large_move"]
    assert len(move_flags) == 1
    assert move_flags[0].obs_date == date(2024, 1, 3)
    assert len(clean) == 2  # row is still present


def test_validate_large_move_not_flagged_below_threshold():
    """A 10% move on an equity_index (threshold=15%) is not flagged."""
    df = _df([("2024-01-02", 100.0), ("2024-01-03", 110.0)])
    clean, flags = validate_provider_output(df, _inst(asset_class=AssetClass.equity_index))
    assert not [f for f in flags if f.rule == "large_move"]


def test_validate_large_move_volatility_threshold():
    """VIX-class instruments: threshold is 200%; a 150% move is not flagged."""
    df = _df([("2024-01-02", 20.0), ("2024-01-03", 50.0)])
    clean, flags = validate_provider_output(df, _inst(asset_class=AssetClass.volatility))
    assert not [f for f in flags if f.rule == "large_move"]


def test_validate_no_large_move_check_on_single_row():
    """Single-row DataFrame: no large_move flags (no prior row to compare to)."""
    df = _df([("2024-01-02", 100.0)])
    _, flags = validate_provider_output(df, _inst())
    assert not [f for f in flags if f.rule == "large_move"]


# ── yfinance signature ────────────────────────────────────────────────────────


def test_yfinance_history_accepts_all_passed_kwargs():
    """Every kwarg _yfinance_fetch passes must be in yfinance's history implementation.

    yfinance.Ticker.history is a *args/**kwargs wrapper; the real parameter list
    lives in PriceHistory.history (yfinance.scrapers.history).
    """
    import inspect

    from yfinance.scrapers.history import PriceHistory

    # These are the kwargs that _yfinance_fetch passes to Ticker.history()
    _PASSED_KWARGS = {"auto_adjust", "start", "end", "period"}

    sig = inspect.signature(PriceHistory.history)
    accepted = set(sig.parameters.keys())
    unknown = _PASSED_KWARGS - accepted
    assert not unknown, (
        f"_yfinance_fetch passes kwargs not accepted by yfinance history: {unknown}. "
        "Update _yfinance_fetch in providers/yahoo.py or pin yfinance to a compatible version."
    )
