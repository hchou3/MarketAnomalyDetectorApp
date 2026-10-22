"""Tests for the instrument registry schema and config/instruments.yaml."""

import pytest

from market_anomaly_detector.registry import (
    Fidelity,
    Instrument,
    SourceEntry,
    ews_symbol_map,
    load_registry,
)

# ── EWS columns that must be covered ──────────────────────────────────────────

EWS_COLUMNS = [
    "XAU BGNL",
    "ECSURPUS",
    "BDIY",
    "CRY",
    "DXY",
    "JPY",
    "GBP",
    "Cl1",
    "VIX",
    "USGG30YR",
    "GT10",
    "USGG2YR",
    "USGG3M",
    "US0001M",
    "GTDEM30Y",
    "GTDEM10Y",
    "GTDEM2Y",
    "EONIA",
    "GTITL30YR",
    "GTITL10YR",
    "GTITL2YR",
    "GTJPY30YR",
    "GTJPY10YR",
    "GTJPY2YR",
    "GTGBP30Y",
    "GTGBP20Y",
    "GTGBP2Y",
    "LUMSTRUU",
    "LMBITR",
    "LUACTRUU",
    "LF98TRUU",
    "LG30TRUU",
    "LP01TREU",
    "EMUSTRUU",
    "LF94TRUU",
    "MXUS",
    "MXEU",
    "MXJP",
    "MXBR",
    "MXRU",
    "MXIN",
    "MXCN",
]

STRESS_IDS = ["vix3m", "vix9d", "vvix", "skew", "ovx", "gvz", "nfci", "stlfsi"]

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def registry() -> list[Instrument]:
    return load_registry()


# ── Structural checks on the live registry ────────────────────────────────────


def test_registry_loads(registry):
    assert len(registry) > 0


def test_ids_are_unique(registry):
    ids = [inst.id for inst in registry]
    assert len(ids) == len(set(ids)), f"Duplicate IDs: {set(i for i in ids if ids.count(i) > 1)}"


def test_priorities_unique_per_instrument(registry):
    for inst in registry:
        priorities = [s.priority for s in inst.sources]
        assert len(priorities) == len(set(priorities)), (
            f"Duplicate priorities in {inst.id!r}: {priorities}"
        )


def test_all_ews_columns_mapped(registry):
    sym_map = ews_symbol_map(registry)
    missing = [col for col in EWS_COLUMNS if col not in sym_map]
    assert not missing, f"EWS columns not in registry: {missing}"


def test_ews_symbols_unique(registry):
    sym_map = ews_symbol_map(registry)
    assert len(sym_map) == len(set(sym_map)), "Duplicate bloomberg_ews symbols"


def test_all_stress_series_mapped(registry):
    ids = {inst.id for inst in registry}
    missing = [sid for sid in STRESS_IDS if sid not in ids]
    assert not missing, f"Stress series not in registry: {missing}"


def test_all_ews_columns_have_exactly_one_instrument(registry):
    from collections import Counter

    ews_symbols: list[str] = []
    for inst in registry:
        for src in inst.sources:
            if src.provider == "bloomberg_ews" and src.symbol:
                ews_symbols.append(src.symbol)

    counts = Counter(ews_symbols)
    dupes = {sym: n for sym, n in counts.items() if n > 1}
    assert not dupes, f"bloomberg_ews symbol appears more than once: {dupes}"


def test_known_providers_only(registry):
    from market_anomaly_detector.registry import KNOWN_PROVIDERS

    for inst in registry:
        for src in inst.sources:
            assert src.provider in KNOWN_PROVIDERS, (
                f"Unknown provider {src.provider!r} on instrument {inst.id!r}"
            )


def test_unavailable_sources_have_no_symbol(registry):
    for inst in registry:
        for src in inst.sources:
            if src.fidelity == Fidelity.unavailable:
                assert src.symbol is None, (
                    f"Instrument {inst.id!r}, provider {src.provider!r}: "
                    f"fidelity=unavailable but symbol={src.symbol!r}"
                )


def test_revised_sources_have_notes(registry):
    for inst in registry:
        for src in inst.sources:
            if src.revised:
                assert src.notes.strip(), (
                    f"Instrument {inst.id!r}, provider {src.provider!r}: "
                    "revised=true but notes is empty"
                )


def test_d5_instruments_have_only_bloomberg_ews_source(registry):
    """LG30TRUU, LP01TREU, LF94TRUU must have no live sources (D5 decision)."""
    d5_ids = {"global_hy_tr", "pan_euro_agg_tr", "lf94_bond_tr"}
    for inst in registry:
        if inst.id in d5_ids:
            live_sources = [s for s in inst.sources if s.provider != "bloomberg_ews"]
            assert not live_sources, (
                f"D5 instrument {inst.id!r} has unexpected live sources: "
                f"{[s.provider for s in live_sources]}"
            )


def test_ews_column_count(registry):
    """All 42 EWS data series are mapped (not counting Y label or Data date)."""
    sym_map = ews_symbol_map(registry)
    ews_covered = [col for col in EWS_COLUMNS if col in sym_map]
    assert len(ews_covered) == 42, f"Expected 42 EWS columns, got {len(ews_covered)}"


# ── Schema-level unit tests (bad YAML fragments) ──────────────────────────────


def _minimal_instrument(**overrides) -> dict:
    base = {
        "id": "test_instr",
        "name": "Test",
        "asset_class": "rate",
        "groups": ["us_rates"],
        "units": "percent",
        "quote": "% per annum",
        "native_frequency": "daily",
        "sources": [
            {"provider": "bloomberg_ews", "symbol": "TEST", "priority": 99, "fidelity": "exact"}
        ],
    }
    base.update(overrides)
    return base


def test_schema_rejects_duplicate_priority():
    with pytest.raises(Exception, match="[Pp]riorit"):
        Instrument.model_validate(
            _minimal_instrument(
                sources=[
                    {
                        "provider": "bloomberg_ews",
                        "symbol": "A",
                        "priority": 1,
                        "fidelity": "exact",
                    },
                    {"provider": "yahoo", "symbol": "B", "priority": 1, "fidelity": "exact"},
                ]
            )
        )


def test_schema_rejects_unknown_provider():
    with pytest.raises(Exception, match="[Uu]nknown provider"):
        SourceEntry.model_validate(
            {"provider": "fake_provider", "symbol": "X", "priority": 1, "fidelity": "exact"}
        )


def test_schema_rejects_unavailable_with_symbol():
    with pytest.raises(Exception, match="unavailable"):
        SourceEntry.model_validate(
            {"provider": "yahoo", "symbol": "SPY", "priority": 1, "fidelity": "unavailable"}
        )


def test_schema_accepts_unavailable_without_symbol():
    src = SourceEntry.model_validate(
        {"provider": "yahoo", "priority": 1, "fidelity": "unavailable"}
    )
    assert src.symbol is None


def test_schema_rejects_revised_without_notes():
    with pytest.raises(Exception, match="[Nn]otes"):
        SourceEntry.model_validate(
            {
                "provider": "fred",
                "symbol": "NFCI",
                "priority": 1,
                "fidelity": "exact",
                "revised": True,
                "notes": "",
            }
        )


def test_schema_rejects_non_snake_case_id():
    with pytest.raises(Exception, match="snake_case"):
        Instrument.model_validate(_minimal_instrument(id="BadId"))


# ── Fix 1: monthly source_frequency needs a realistic lag ─────────────────────


def test_schema_rejects_monthly_source_with_short_lag():
    with pytest.raises(Exception, match="28"):
        SourceEntry.model_validate(
            {
                "provider": "fred",
                "symbol": "IRLTLT01DEM156N",
                "priority": 1,
                "fidelity": "proxy",
                "source_frequency": "monthly",
                "publication_lag_days": 1,
            }
        )


def test_schema_accepts_monthly_source_with_adequate_lag():
    src = SourceEntry.model_validate(
        {
            "provider": "fred",
            "symbol": "IRLTLT01DEM156N",
            "priority": 1,
            "fidelity": "proxy",
            "source_frequency": "monthly",
            "publication_lag_days": 40,
        }
    )
    assert src.source_frequency.value == "monthly"


def test_monthly_fred_proxies_have_source_frequency_and_adequate_lag(registry):
    """de_10y_yield, it_10y_yield, jp_10y_yield FRED sources must not be treated as daily."""
    monthly_instrument_ids = {"de_10y_yield", "it_10y_yield", "jp_10y_yield"}
    for inst in registry:
        if inst.id not in monthly_instrument_ids:
            continue
        fred_src = next((s for s in inst.sources if s.provider == "fred"), None)
        assert fred_src is not None, f"{inst.id} has no FRED source"
        assert (
            fred_src.source_frequency is not None and fred_src.source_frequency.value == "monthly"
        ), f"{inst.id} FRED source missing source_frequency=monthly"
        assert fred_src.publication_lag_days >= 28, (
            f"{inst.id} FRED source publication_lag_days={fred_src.publication_lag_days} is too short for monthly data"
        )


# ── Fix 2: extra="forbid" catches YAML typos ──────────────────────────────────


def test_schema_rejects_unknown_source_field():
    with pytest.raises(Exception):
        SourceEntry.model_validate(
            {
                "provider": "fred",
                "symbol": "DGS10",
                "priority": 1,
                "fidelity": "exact",
                "fedelity": "exact",
            }
        )


def test_schema_rejects_unknown_instrument_field():
    with pytest.raises(Exception):
        Instrument.model_validate(_minimal_instrument(typo_field="oops"))


# ── Fix 3: FRED FX lag and FX priority ────────────────────────────────────────


def test_fred_fx_lag_at_least_ten_days(registry):
    """usd_jpy and gbp_usd FRED sources use a conservative 10-day floor.

    ALFRED-measured typical H.10 lag is 1–3 calendar days, but 10 days is used
    as a guaranteed safe floor to cover holiday weekends and any publication
    delays without risking look-ahead bias in backtests.
    """
    fx_ids = {"usd_jpy", "gbp_usd"}
    for inst in registry:
        if inst.id not in fx_ids:
            continue
        fred_src = next((s for s in inst.sources if s.provider == "fred"), None)
        assert fred_src is not None, f"{inst.id} has no FRED source"
        assert fred_src.publication_lag_days >= 10, (
            f"{inst.id} FRED lag={fred_src.publication_lag_days}; "
            "must be >= 10 (conservative floor for H.10 FX release)"
        )


def test_yahoo_is_priority_1_for_fx(registry):
    """Yahoo must be the primary (priority 1) source for usd_jpy and gbp_usd.

    FRED H.10 FX data carries a 10-day lag; Yahoo provides same-day spot rates
    with no meaningful delay and should be used for live ingestion.
    """
    fx_ids = {"usd_jpy", "gbp_usd"}
    for inst in registry:
        if inst.id not in fx_ids:
            continue
        yahoo_src = next((s for s in inst.sources if s.provider == "yahoo"), None)
        assert yahoo_src is not None, f"{inst.id} has no Yahoo source"
        assert yahoo_src.priority == 1, (
            f"{inst.id} Yahoo source has priority={yahoo_src.priority}; expected 1"
        )


# ── Step 4: auto_adjust enforcement for Yahoo ETFs ────────────────────────────

_YAHOO_ETF_SYMBOLS = {
    # Original ETFs
    "DBC",
    "MBB",
    "MUB",
    "LQD",
    "HYG",
    "EMB",
    "VGK",
    "EWJ",
    "EWZ",
    "INDA",
    "MCHI",
    # Longer-history proxy ETFs / mutual funds added for backward splicing
    "VWEHX",  # Vanguard HY Corporate Bond mutual fund (pre-HYG)
    "IEV",  # iShares Europe ETF (pre-VGK)
    "EPI",  # WisdomTree India Earnings ETF (pre-INDA)
    "FXI",  # iShares China Large-Cap ETF (pre-MCHI)
}


def test_yahoo_etf_sources_have_auto_adjust(registry):
    """ETF proxies on Yahoo must set auto_adjust=True to get total-return prices.

    These ETFs distribute dividends that would otherwise make their price series
    non-comparable to the Bloomberg total-return indices they proxy.
    """
    for inst in registry:
        for src in inst.sources:
            if src.provider == "yahoo" and src.symbol in _YAHOO_ETF_SYMBOLS:
                assert src.auto_adjust, (
                    f"Instrument {inst.id!r}: Yahoo ETF source {src.symbol!r} "
                    "must have auto_adjust=True for total-return prices"
                )
