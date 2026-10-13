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
