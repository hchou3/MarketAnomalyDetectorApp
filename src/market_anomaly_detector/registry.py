"""
Instrument registry: Pydantic schema and loader for config/instruments.yaml.

Use load_registry() to get the validated list of Instrument objects.
Use ews_symbol_map() to get a {bloomberg_ews_symbol: instrument_id} mapping.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .paths import REPO_ROOT

_CONFIG_PATH = REPO_ROOT / "config" / "instruments.yaml"

# ── Enumerations ───────────────────────────────────────────────────────────────

KNOWN_PROVIDERS = {"fred", "yahoo", "cboe", "bloomberg_ews"}


class AssetClass(str, Enum):
    fx = "fx"
    rate = "rate"
    equity_index = "equity_index"
    commodity = "commodity"
    volatility = "volatility"
    credit = "credit"
    macro_index = "macro_index"


class Units(str, Enum):
    price = "price"
    percent = "percent"
    index_points = "index_points"


class Frequency(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"


class Fidelity(str, Enum):
    exact = "exact"
    proxy = "proxy"
    unavailable = "unavailable"


# ── Source entry ───────────────────────────────────────────────────────────────


class SourceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    symbol: Optional[str] = None
    priority: int
    fidelity: Fidelity
    # When the source's native frequency differs from the instrument's (e.g. a
    # monthly FRED series for an otherwise-daily instrument), set this explicitly.
    # None means "same as the instrument's native_frequency".
    source_frequency: Optional[Frequency] = None
    publication_lag_days: int = 0
    revised: bool = False
    # Yahoo-specific: True folds splits and dividends into Close (total-return).
    # Set True for ETF proxies; leave False for indices, futures, and FX pairs.
    auto_adjust: bool = False
    # False when the live series and the EWS reference are on different index
    # scales (e.g. ^GSPC level ~5500 vs Bloomberg MXUS ~600); suppresses the
    # level-ratio check even for exact-fidelity sources.
    compare_level: bool = True
    notes: str = ""

    @field_validator("provider")
    @classmethod
    def provider_is_known(cls, v: str) -> str:
        if v not in KNOWN_PROVIDERS:
            raise ValueError(f"Unknown provider {v!r}. Known: {sorted(KNOWN_PROVIDERS)}")
        return v

    @model_validator(mode="after")
    def unavailable_has_no_symbol(self) -> SourceEntry:
        if self.fidelity == Fidelity.unavailable and self.symbol is not None:
            raise ValueError(
                f"Source with fidelity 'unavailable' must not have a symbol, got {self.symbol!r}"
            )
        return self

    @model_validator(mode="after")
    def revised_requires_notes(self) -> SourceEntry:
        if self.revised and not self.notes.strip():
            raise ValueError(
                "revised: true requires a non-empty notes field"
                " explaining how revisions are handled"
            )
        return self

    @model_validator(mode="after")
    def monthly_source_needs_realistic_lag(self) -> SourceEntry:
        if self.source_frequency == Frequency.monthly and self.publication_lag_days < 28:
            raise ValueError(
                f"A monthly source must have publication_lag_days >= 28 "
                f"(got {self.publication_lag_days}). Monthly series typically "
                f"lag 30–40 days after the observation month end."
            )
        return self


# ── Instrument ─────────────────────────────────────────────────────────────────


class Instrument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    asset_class: AssetClass
    groups: list[str]
    units: Units
    quote: str
    native_frequency: Frequency
    sources: list[SourceEntry]
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    notes: str = ""
    # For proxy instruments: the canonical instrument whose EWS column this
    # proxies.  None for all primary instruments.
    reference_instrument: Optional[str] = None
    # Set False to exclude this instrument from live ingestion and staleness checks.
    active: bool = True
    # Set True for instruments whose values are legitimately negative (e.g.
    # economic surprise indices, rate spreads, real yields).  Suppresses the
    # negative_value validation flag.
    allow_negative: bool = False

    @field_validator("id")
    @classmethod
    def id_is_snake_case(cls, v: str) -> str:
        import re

        if not re.match(r"^[a-z][a-z0-9_]*$", v):
            raise ValueError(f"id must be lowercase snake_case, got {v!r}")
        return v

    @model_validator(mode="after")
    def priorities_unique(self) -> Instrument:
        priorities = [s.priority for s in self.sources]
        if len(priorities) != len(set(priorities)):
            dupes = [p for p in priorities if priorities.count(p) > 1]
            raise ValueError(
                f"Source priorities must be unique within instrument {self.id!r}; "
                f"duplicate priority values: {sorted(set(dupes))}"
            )
        return self

    @model_validator(mode="after")
    def providers_unique(self) -> Instrument:
        providers = [s.provider for s in self.sources]
        if len(providers) != len(set(providers)):
            from collections import Counter

            dupes = sorted({p for p, n in Counter(providers).items() if n > 1})
            raise ValueError(
                f"Instrument {self.id!r} has multiple sources with the same provider: "
                f"{dupes}. Each provider must appear at most once per instrument; "
                "use a separate proxy instrument for an alternative symbol."
            )
        return self


# ── Registry loader ────────────────────────────────────────────────────────────


def load_registry(path: Path = _CONFIG_PATH) -> list[Instrument]:
    """Load, parse, and validate the instrument registry YAML.

    Raises ValueError if any cross-instrument constraint is violated.
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    instruments = [Instrument.model_validate(item) for item in raw]

    # Cross-instrument: IDs must be unique
    ids = [inst.id for inst in instruments]
    if len(ids) != len(set(ids)):
        from collections import Counter

        dupes = [id_ for id_, n in Counter(ids).items() if n > 1]
        raise ValueError(f"Duplicate instrument IDs: {dupes}")

    # Cross-instrument: bloomberg_ews symbols must each appear exactly once
    ews_symbols: list[str] = []
    for inst in instruments:
        for src in inst.sources:
            if src.provider == "bloomberg_ews":
                if src.symbol:
                    ews_symbols.append(src.symbol)

    if len(ews_symbols) != len(set(ews_symbols)):
        from collections import Counter

        dupes = [sym for sym, n in Counter(ews_symbols).items() if n > 1]
        raise ValueError(f"bloomberg_ews symbol appears in multiple instruments: {dupes}")

    # Cross-instrument: reference_instrument must name an existing instrument ID
    all_ids = {inst.id for inst in instruments}
    for inst in instruments:
        if inst.reference_instrument and inst.reference_instrument not in all_ids:
            raise ValueError(
                f"Instrument {inst.id!r} has reference_instrument "
                f"{inst.reference_instrument!r} which does not exist in the registry"
            )

    return instruments


def ews_symbol_map(instruments: list[Instrument] | None = None) -> dict[str, str]:
    """Return {bloomberg_ews_symbol: instrument_id} for all mapped EWS columns."""
    if instruments is None:
        instruments = load_registry()
    result: dict[str, str] = {}
    for inst in instruments:
        for src in inst.sources:
            if src.provider == "bloomberg_ews" and src.symbol:
                result[src.symbol] = inst.id
    return result
