"""Feature specification for the Market Anomaly Detector.

A FeatureSpec describes one output column: which instrument provides its input,
what transform to apply, and a human-readable label for explanations.

Transforms
----------
log_return  : log(P[t]) - log(P[t-w])                  (price / index_points)
yield_change: (P[t] - P[t-w]) * 100  in basis points   (percent)
drawdown    : log(P[t]) - max(log(P[t-w:t]))            (price / index_points)
rvol        : annualised std of daily log returns        (window = trading days)
z_score     : z-score of 1-week return/change over trailing window (all units)
spread      : (P1[t] - P2[t]) * 100  in basis points    (percent, two instruments)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSpec:
    col: str
    label: str
    instrument_id: str
    units: str
    transform: str
    window: int
    instrument_id_2: str = field(default="")
    # Optional list of proxy instrument IDs to extend this instrument's history.
    # load_prices splices them in ascending date order before the primary's first row,
    # using only entries rated "proxy" or "exact" in the coverage report.
    fallback_instrument_ids: list[str] = field(default_factory=list)


def spec_hash(spec: list[FeatureSpec]) -> str:
    """SHA-256 of the FeatureSpec content (col, instrument_id, instrument_id_2,
    transform, window, fallback_instrument_ids).  label and units are excluded
    so cosmetic renames don't invalidate the hash.  Used to detect accidental
    edits to a frozen spec (e.g. SPEC_V1).
    """
    h = hashlib.sha256()
    for feat in spec:
        row = "|".join(
            [
                feat.col,
                feat.instrument_id,
                feat.instrument_id_2,
                feat.transform,
                str(feat.window),
                ",".join(feat.fallback_instrument_ids),
            ]
        )
        h.update((row + "\n").encode("utf-8"))
    return h.hexdigest()


def validate_spec_fallbacks(spec: list[FeatureSpec], instruments: list) -> None:
    """Raise ValueError if any fallback_instrument_id doesn't exist in instruments.

    Parameters
    ----------
    spec:
        Feature specification list to validate.
    instruments:
        Registry instrument list (each must have an ``.id`` attribute).
    """
    inst_ids = {inst.id for inst in instruments}
    for feat in spec:
        for fb_id in feat.fallback_instrument_ids:
            if fb_id not in inst_ids:
                raise ValueError(
                    f"FeatureSpec {feat.col!r}: fallback_instrument_id {fb_id!r} "
                    f"not found in instrument registry"
                )


# ── Feature set v1 ────────────────────────────────────────────────────────────
# Live providers only; no bloomberg_ews inputs.
# Instruments: vix, msci_usa, dollar_index, usd_jpy, gbp_usd, gold, wti_front,
#              us_10y_yield, us_2y_yield, us_3m_yield, skew
# NFCI / STLFSI excluded from v1 (revised series, decision F1).

SPEC_V1: list[FeatureSpec] = [
    # ── VIX (index_points) ────────────────────────────────────────────────────
    FeatureSpec("vix_lr_1w", "VIX, 1-week log return", "vix", "index_points", "log_return", 1),
    FeatureSpec("vix_lr_4w", "VIX, 4-week log return", "vix", "index_points", "log_return", 4),
    FeatureSpec("vix_lr_12w", "VIX, 12-week log return", "vix", "index_points", "log_return", 12),
    FeatureSpec("vix_rvol_20d", "VIX, realized vol 20-day", "vix", "index_points", "rvol", 20),
    FeatureSpec(
        "vix_z_1w", "VIX, z-score of 1w return (156w)", "vix", "index_points", "z_score", 156
    ),
    # ── S&P 500 / MSCI USA (index_points) ────────────────────────────────────
    FeatureSpec(
        "sp500_lr_1w", "S&P 500, 1-week log return", "msci_usa", "index_points", "log_return", 1
    ),
    FeatureSpec(
        "sp500_lr_4w", "S&P 500, 4-week log return", "msci_usa", "index_points", "log_return", 4
    ),
    FeatureSpec(
        "sp500_lr_12w", "S&P 500, 12-week log return", "msci_usa", "index_points", "log_return", 12
    ),
    FeatureSpec(
        "sp500_rvol_20d", "S&P 500, realized vol 20-day", "msci_usa", "index_points", "rvol", 20
    ),
    FeatureSpec(
        "sp500_drawdown_52w",
        "S&P 500, 52-week drawdown",
        "msci_usa",
        "index_points",
        "drawdown",
        52,
    ),
    FeatureSpec(
        "sp500_z_1w",
        "S&P 500, z-score of 1w return (156w)",
        "msci_usa",
        "index_points",
        "z_score",
        156,
    ),
    # ── Dollar Index (index_points) ───────────────────────────────────────────
    FeatureSpec(
        "dxy_lr_1w",
        "Dollar Index, 1-week log return",
        "dollar_index",
        "index_points",
        "log_return",
        1,
    ),
    FeatureSpec(
        "dxy_lr_4w",
        "Dollar Index, 4-week log return",
        "dollar_index",
        "index_points",
        "log_return",
        4,
    ),
    FeatureSpec(
        "dxy_lr_12w",
        "Dollar Index, 12-week log return",
        "dollar_index",
        "index_points",
        "log_return",
        12,
    ),
    FeatureSpec(
        "dxy_z_1w",
        "Dollar Index, z-score of 1w return (156w)",
        "dollar_index",
        "index_points",
        "z_score",
        156,
    ),
    # ── USD/JPY (price) ───────────────────────────────────────────────────────
    FeatureSpec("usdjpy_lr_1w", "USD/JPY, 1-week log return", "usd_jpy", "price", "log_return", 1),
    FeatureSpec("usdjpy_lr_4w", "USD/JPY, 4-week log return", "usd_jpy", "price", "log_return", 4),
    FeatureSpec(
        "usdjpy_lr_12w", "USD/JPY, 12-week log return", "usd_jpy", "price", "log_return", 12
    ),
    FeatureSpec(
        "usdjpy_z_1w", "USD/JPY, z-score of 1w return (156w)", "usd_jpy", "price", "z_score", 156
    ),
    # ── GBP/USD (price) ───────────────────────────────────────────────────────
    FeatureSpec("gbpusd_lr_1w", "GBP/USD, 1-week log return", "gbp_usd", "price", "log_return", 1),
    FeatureSpec("gbpusd_lr_4w", "GBP/USD, 4-week log return", "gbp_usd", "price", "log_return", 4),
    FeatureSpec(
        "gbpusd_lr_12w", "GBP/USD, 12-week log return", "gbp_usd", "price", "log_return", 12
    ),
    FeatureSpec(
        "gbpusd_z_1w", "GBP/USD, z-score of 1w return (156w)", "gbp_usd", "price", "z_score", 156
    ),
    # ── Gold (price) ──────────────────────────────────────────────────────────
    FeatureSpec("gold_lr_1w", "Gold, 1-week log return", "gold", "price", "log_return", 1),
    FeatureSpec("gold_lr_4w", "Gold, 4-week log return", "gold", "price", "log_return", 4),
    FeatureSpec("gold_lr_12w", "Gold, 12-week log return", "gold", "price", "log_return", 12),
    FeatureSpec("gold_rvol_20d", "Gold, realized vol 20-day", "gold", "price", "rvol", 20),
    FeatureSpec("gold_z_1w", "Gold, z-score of 1w return (156w)", "gold", "price", "z_score", 156),
    # ── WTI crude (price) ─────────────────────────────────────────────────────
    FeatureSpec("wti_lr_1w", "WTI crude, 1-week log return", "wti_front", "price", "log_return", 1),
    FeatureSpec("wti_lr_4w", "WTI crude, 4-week log return", "wti_front", "price", "log_return", 4),
    FeatureSpec(
        "wti_lr_12w", "WTI crude, 12-week log return", "wti_front", "price", "log_return", 12
    ),
    FeatureSpec("wti_rvol_20d", "WTI crude, realized vol 20-day", "wti_front", "price", "rvol", 20),
    FeatureSpec(
        "wti_z_1w", "WTI crude, z-score of 1w return (156w)", "wti_front", "price", "z_score", 156
    ),
    # ── US 10-year yield (percent) ────────────────────────────────────────────
    FeatureSpec(
        "us10y_yc_1w",
        "US 10Y yield, 1-week change (bp)",
        "us_10y_yield",
        "percent",
        "yield_change",
        1,
    ),
    FeatureSpec(
        "us10y_yc_4w",
        "US 10Y yield, 4-week change (bp)",
        "us_10y_yield",
        "percent",
        "yield_change",
        4,
    ),
    FeatureSpec(
        "us10y_yc_12w",
        "US 10Y yield, 12-week change (bp)",
        "us_10y_yield",
        "percent",
        "yield_change",
        12,
    ),
    FeatureSpec(
        "us10y_z_1w",
        "US 10Y yield, z-score of 1w change (156w)",
        "us_10y_yield",
        "percent",
        "z_score",
        156,
    ),
    # ── US 2-year yield (percent) ─────────────────────────────────────────────
    FeatureSpec(
        "us2y_yc_1w", "US 2Y yield, 1-week change (bp)", "us_2y_yield", "percent", "yield_change", 1
    ),
    FeatureSpec(
        "us2y_yc_4w", "US 2Y yield, 4-week change (bp)", "us_2y_yield", "percent", "yield_change", 4
    ),
    FeatureSpec(
        "us2y_yc_12w",
        "US 2Y yield, 12-week change (bp)",
        "us_2y_yield",
        "percent",
        "yield_change",
        12,
    ),
    FeatureSpec(
        "us2y_z_1w",
        "US 2Y yield, z-score of 1w change (156w)",
        "us_2y_yield",
        "percent",
        "z_score",
        156,
    ),
    # ── US 3-month yield (percent) ────────────────────────────────────────────
    FeatureSpec(
        "us3m_yc_1w", "US 3M yield, 1-week change (bp)", "us_3m_yield", "percent", "yield_change", 1
    ),
    FeatureSpec(
        "us3m_yc_4w", "US 3M yield, 4-week change (bp)", "us_3m_yield", "percent", "yield_change", 4
    ),
    FeatureSpec(
        "us3m_yc_12w",
        "US 3M yield, 12-week change (bp)",
        "us_3m_yield",
        "percent",
        "yield_change",
        12,
    ),
    FeatureSpec(
        "us3m_z_1w",
        "US 3M yield, z-score of 1w change (156w)",
        "us_3m_yield",
        "percent",
        "z_score",
        156,
    ),
    # ── Yield spreads (percent → basis points) ────────────────────────────────
    FeatureSpec(
        "spread_10y_3m",
        "US 10Y-3M yield spread (bp)",
        "us_10y_yield",
        "percent",
        "spread",
        0,
        instrument_id_2="us_3m_yield",
    ),
    FeatureSpec(
        "spread_10y_2y",
        "US 10Y-2Y yield spread (bp)",
        "us_10y_yield",
        "percent",
        "spread",
        0,
        instrument_id_2="us_2y_yield",
    ),
    # ── CBOE SKEW (index_points) ──────────────────────────────────────────────
    FeatureSpec(
        "skew_lr_1w", "CBOE SKEW, 1-week log return", "skew", "index_points", "log_return", 1
    ),
    FeatureSpec(
        "skew_lr_4w", "CBOE SKEW, 4-week log return", "skew", "index_points", "log_return", 4
    ),
    FeatureSpec(
        "skew_lr_12w", "CBOE SKEW, 12-week log return", "skew", "index_points", "log_return", 12
    ),
    FeatureSpec(
        "skew_z_1w",
        "CBOE SKEW, z-score of 1w return (156w)",
        "skew",
        "index_points",
        "z_score",
        156,
    ),
]


# ── Feature set v2 ────────────────────────────────────────────────────────────
# Extends v1 with FRED fallback splicing for GBP/USD and USD/JPY.
# GBP/USD + USD/JPY: FRED H.10 releases carry history back to 1971;
# spliced as proxy before Yahoo's reliable start (~2003-12-01).

SPEC_V2: list[FeatureSpec] = [
    feat
    if feat.col not in {"gbpusd_lr_1w", "gbpusd_lr_4w", "gbpusd_lr_12w", "gbpusd_z_1w"}
    else FeatureSpec(
        feat.col,
        feat.label,
        feat.instrument_id,
        feat.units,
        feat.transform,
        feat.window,
        feat.instrument_id_2,
        fallback_instrument_ids=["gbp_usd_fred"],
    )
    for feat in SPEC_V1
]
