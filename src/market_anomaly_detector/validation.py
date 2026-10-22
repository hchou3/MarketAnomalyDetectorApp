"""Pre-write validation for provider DataFrames.

Rules applied in order:
  1. Non-numeric values — dropped silently (providers should already filter these).
  2. Duplicate obs_dates — exact duplicates deduplicated to first; conflicting
     duplicates (same date, different value) flagged as ``duplicate_dates_disagree``.
  3. Negative values — flagged as ``negative_value`` only for price/index_points
     instruments where the instrument does not set ``allow_negative=True``.
     Percent-unit instruments (yields, rates, spreads) are always exempt.
  4. Large single-day moves — flagged as ``large_move``.  For price/index_points
     series, abs(pct_change) is compared against a per-asset-class relative
     threshold.  For percent-unit series, the absolute change (in percentage
     points, i.e. basis points / 100) is compared against a bp threshold.

Flagged rows are *still written*; flags are advisory quality markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

# Units for which negative values are expected (yields, rates, spreads, etc.)
_PERCENT_UNITS: frozenset[str] = frozenset({"percent"})

# abs(pct_change) threshold for price/index_points instruments
_LARGE_MOVE_THRESHOLDS: dict[str, float] = {
    "equity_index": 0.15,
    "fx": 0.10,
    "rate": 0.50,
    "commodity": 0.25,
    "volatility": 2.00,
    "credit": 0.20,
    "macro_index": 0.50,
}
_DEFAULT_LARGE_MOVE_THRESHOLD = 0.50

# Absolute change threshold in percentage points for percent-unit instruments.
# 2 pp = 200 bp; a daily move larger than this is flagged for any rate/yield.
_LARGE_MOVE_BP_THRESHOLD_PP: float = 2.0  # percentage points (= 200 bp)


@dataclass
class ValidationFlag:
    """A single data-quality observation for one (instrument, provider, obs_date)."""

    obs_date: date
    rule: str  # "duplicate_dates_disagree" | "negative_value" | "large_move"
    detail: str


def validate_provider_output(
    df: pd.DataFrame,
    instrument: Any,
) -> tuple[pd.DataFrame, list[ValidationFlag]]:
    """Validate and clean a provider DataFrame before writing to the price store.

    Parameters
    ----------
    df:
        DataFrame with columns ``obs_date`` (date) and ``value`` (numeric).
    instrument:
        Registry ``Instrument`` object; used for asset_class and id checks.

    Returns
    -------
    (cleaned_df, flags)
        *cleaned_df* has unique obs_dates and all non-numeric rows removed.
        *flags* lists quality issues found (rows are still in cleaned_df).
    """
    flags: list[ValidationFlag] = []

    df = df.copy()

    # ── 1. Drop non-numeric values ────────────────────────────────────────────
    numeric_vals = pd.to_numeric(df["value"], errors="coerce")
    df = df[numeric_vals.notna()].copy()
    if df.empty:
        return df.reset_index(drop=True), flags

    # ── 2. Duplicate obs_dates ────────────────────────────────────────────────
    duped = df[df.duplicated("obs_date", keep=False)]
    if not duped.empty:
        for obs_d in duped["obs_date"].unique():
            group = df[df["obs_date"] == obs_d]
            unique_vals = group["value"].unique()
            if len(unique_vals) > 1:
                flags.append(
                    ValidationFlag(
                        obs_date=obs_d,
                        rule="duplicate_dates_disagree",
                        detail=f"values: {sorted(float(v) for v in unique_vals)}",
                    )
                )
        df = df.drop_duplicates(subset="obs_date", keep="first")

    # ── 3. Negative values ────────────────────────────────────────────────────
    # Only flag for price/index_points instruments unless allow_negative is set.
    units = instrument.units.value if hasattr(instrument.units, "value") else str(instrument.units)
    asset_class = (
        instrument.asset_class.value
        if hasattr(instrument.asset_class, "value")
        else str(instrument.asset_class)
    )
    allow_negative = getattr(instrument, "allow_negative", False)

    if units not in _PERCENT_UNITS and not allow_negative:
        neg_rows = df[df["value"] < 0]
        for _, row in neg_rows.iterrows():
            flags.append(
                ValidationFlag(
                    obs_date=row["obs_date"],
                    rule="negative_value",
                    detail=f"value={float(row['value']):.6g}",
                )
            )

    # ── 4. Large single-day moves ─────────────────────────────────────────────
    if len(df) > 1:
        sorted_df = df.sort_values("obs_date").reset_index(drop=True)
        if units in _PERCENT_UNITS:
            # For yield/rate series use absolute change in percentage points.
            abs_changes = sorted_df["value"].diff().abs()
            for idx in abs_changes[abs_changes > _LARGE_MOVE_BP_THRESHOLD_PP].index:
                if idx > 0:
                    obs_d = sorted_df.loc[idx, "obs_date"]
                    val = float(sorted_df.loc[idx, "value"])
                    prev_val = float(sorted_df.loc[idx - 1, "value"])
                    chg_bp = float(abs_changes.loc[idx]) * 100  # pp → bp
                    flags.append(
                        ValidationFlag(
                            obs_date=obs_d,
                            rule="large_move",
                            detail=f"abs_change={chg_bp:.0f}bp ({prev_val:.6g} -> {val:.6g})",
                        )
                    )
        else:
            threshold = _LARGE_MOVE_THRESHOLDS.get(asset_class, _DEFAULT_LARGE_MOVE_THRESHOLD)
            pct_changes = sorted_df["value"].pct_change().abs()
            for idx in pct_changes[pct_changes > threshold].index:
                if idx > 0:
                    obs_d = sorted_df.loc[idx, "obs_date"]
                    val = float(sorted_df.loc[idx, "value"])
                    prev_val = float(sorted_df.loc[idx - 1, "value"])
                    pct = float(pct_changes.loc[idx])
                    flags.append(
                        ValidationFlag(
                            obs_date=obs_d,
                            rule="large_move",
                            detail=f"pct_change={pct:.1%} ({prev_val:.6g} -> {val:.6g})",
                        )
                    )

    return df.reset_index(drop=True), flags
