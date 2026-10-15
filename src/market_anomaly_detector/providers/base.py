"""Provider protocol and Tick type for the data ingestion layer."""

from __future__ import annotations

from datetime import date
from typing import Iterator, NamedTuple, Protocol

import pandas as pd


class Tick(NamedTuple):
    """A single real-time data point (Phase 8 subscribe interface)."""

    symbol: str
    ts: date
    value: float


class Provider(Protocol):
    """Interface all data providers implement.

    ``get_history`` is the batch backfill / update path.
    ``subscribe`` is the real-time streaming path (Phase 8; raises
    NotImplementedError until then).
    """

    name: str

    def get_history(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Return DataFrame[obs_date: date, value: float].

        Sorted ascending by obs_date; no duplicate dates; no NaN values.
        Raises ValueError when the symbol returns no usable data.
        """
        ...

    def subscribe(self, symbols: list[str]) -> Iterator[Tick]:
        """Real-time streaming (Phase 8 stub)."""
        raise NotImplementedError
