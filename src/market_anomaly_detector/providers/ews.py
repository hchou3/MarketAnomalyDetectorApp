"""EwsCsvProvider — reads Bloomberg EWS weekly data from the historical CSV.

The CSV layout: Y (label), Data (M/D/YYYY date), then one column per EWS series.
known_at = obs_date for every row (the EWS convention: the file value is already
known on the Tuesday it is dated; there is no publication lag).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

from ..paths import DATA_PATH

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class EwsCsvProvider:
    """Read Bloomberg EWS weekly data from the historical CSV file."""

    name = "bloomberg_ews"

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self._path = Path(path) if path is not None else DATA_PATH
        self._df: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self._path, parse_dates=["Data"])
            self._df = df.sort_values("Data").reset_index(drop=True)
        return self._df

    def series_columns(self) -> list[str]:
        """Return the EWS series column names (all columns except Y and Data)."""
        return [c for c in self._load().columns if c not in ("Y", "Data")]

    def get_history(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
    ) -> pd.DataFrame:
        """Return DataFrame[obs_date, value] for one EWS series; NaN rows dropped."""
        df = self._load()
        if symbol not in df.columns:
            raise KeyError(
                f"EWS symbol {symbol!r} not found. "
                f"Available columns: {[c for c in df.columns if c not in ('Y', 'Data')]}"
            )
        result = df[["Data", symbol]].rename(columns={"Data": "obs_date", symbol: "value"})
        result = result.dropna(subset=["value"]).copy()
        result["obs_date"] = result["obs_date"].dt.date
        if start is not None:
            result = result[result["obs_date"] >= start]
        if end is not None:
            result = result[result["obs_date"] <= end]
        return result.reset_index(drop=True)

    def get_labels(self) -> pd.DataFrame:
        """Return DataFrame[week, value] from the Y column; NaN rows dropped."""
        df = self._load()
        result = df[["Data", "Y"]].rename(columns={"Data": "week", "Y": "value"})
        result = result.dropna(subset=["value"]).copy()
        result["week"] = result["week"].dt.date
        result["value"] = result["value"].astype(int)
        return result.reset_index(drop=True)


def load_ews(
    session: Session,
    csv_path: Optional[Path | str] = None,
    instruments: Optional[list] = None,
) -> dict:
    """Load all EWS series into prices and Y into labels.

    Syncs the instrument registry first, then upserts bloomberg_ews price rows
    (known_at = obs_date) and ews_v1 label rows.

    Does NOT commit; the caller is responsible for committing or rolling back.

    Returns a dict with keys:
        prices_written:    price rows inserted or updated across all series
        labels_written:    label rows inserted or updated
        instruments_ok:    number of EWS series loaded without error
        instruments_failed: number of EWS series that raised an error
    """
    from ..db.repository import (
        add_run_item,
        finish_run,
        start_run,
        sync_registry,
        upsert_labels,
        upsert_prices,
    )
    from ..registry import ews_symbol_map, load_registry

    if instruments is None:
        instruments = load_registry()

    provider = EwsCsvProvider(path=csv_path)
    sym_map = ews_symbol_map(instruments)  # {ews_symbol: instrument_id}

    sync_registry(session, instruments)
    session.flush()

    run = start_run(session, mode="load_ews")
    session.flush()

    total_prices = 0
    ok = 0
    failed = 0

    for ews_symbol, instrument_id in sorted(sym_map.items()):
        try:
            hist = provider.get_history(ews_symbol)
            if hist.empty:
                add_run_item(session, run, instrument_id, "bloomberg_ews", "ok")
                ok += 1
                continue

            rows = [
                {
                    "instrument_id": instrument_id,
                    "provider": "bloomberg_ews",
                    "obs_date": row.obs_date,
                    "value": row.value,
                    "known_at": row.obs_date,
                    "run_id": run.id,
                }
                for row in hist.itertuples(index=False)
            ]
            n = upsert_prices(session, rows)
            add_run_item(
                session,
                run,
                instrument_id,
                "bloomberg_ews",
                "ok",
                rows_written=n,
                first_date=rows[0]["obs_date"],
                last_date=rows[-1]["obs_date"],
            )
            total_prices += n
            ok += 1
        except Exception as exc:
            add_run_item(session, run, instrument_id, "bloomberg_ews", "error", error=str(exc))
            failed += 1

    # Load Y column into labels as ews_v1
    labels_df = provider.get_labels()
    label_rows = [
        {
            "label_set": "ews_v1",
            "week": row.week,
            "value": row.value,
            "source": "bloomberg_ews",
        }
        for row in labels_df.itertuples(index=False)
    ]
    n_labels = upsert_labels(session, label_rows)
    session.flush()

    finish_run(
        session,
        run,
        status="success" if failed == 0 else "partial",
        instruments_ok=ok,
        instruments_failed=failed,
        rows_written=total_prices,
    )
    session.flush()

    return {
        "prices_written": total_prices,
        "labels_written": n_labels,
        "instruments_ok": ok,
        "instruments_failed": failed,
    }
