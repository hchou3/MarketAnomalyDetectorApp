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
        n = n_new = n_revised = 0
        first_date = last_date = None
        error_msg = None
        try:
            with session.begin_nested():  # SAVEPOINT — rolls back only this series on error
                hist = provider.get_history(ews_symbol)
                if not hist.empty:
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
                    n_new, n_revised = upsert_prices(session, rows)
                    n = n_new + n_revised
                    first_date = rows[0]["obs_date"]
                    last_date = rows[-1]["obs_date"]
        except Exception as exc:
            error_msg = str(exc)[:500]

        # Record the outcome outside the savepoint so it is always persisted
        # even when the series write failed.
        if error_msg is None:
            add_run_item(
                session,
                run,
                instrument_id,
                "bloomberg_ews",
                "ok",
                rows_new=n_new,
                rows_revised=n_revised,
                first_date=first_date,
                last_date=last_date,
            )
            total_prices += n
            ok += 1
        else:
            add_run_item(session, run, instrument_id, "bloomberg_ews", "error", error=error_msg)
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


def rebuild_ews_frame(
    session: Session,
    instruments: Optional[list] = None,
) -> pd.DataFrame:
    """Rebuild a DataFrame matching load_data() from bloomberg_ews prices and ews_v1 labels.

    Returns columns: Data (datetime64[ns]), Y (int), and the 42 EWS series (float64),
    sorted by date — identical to load_data(DATA_PATH) when populated via load_ews().
    """
    from sqlalchemy import select

    from ..db.models import LabelRow, PriceRow
    from ..registry import ews_symbol_map, load_registry

    if instruments is None:
        instruments = load_registry()

    sym_map = ews_symbol_map(instruments)
    id_to_sym = {v: k for k, v in sym_map.items()}

    price_rows = session.execute(
        select(PriceRow.obs_date, PriceRow.instrument_id, PriceRow.value).where(
            PriceRow.provider == "bloomberg_ews"
        )
    ).all()
    price_df = pd.DataFrame(
        [(r.obs_date, id_to_sym[r.instrument_id], float(r.value)) for r in price_rows],
        columns=["obs_date", "symbol", "value"],
    )
    wide = price_df.pivot(index="obs_date", columns="symbol", values="value")
    wide.columns.name = None

    label_rows = session.execute(
        select(LabelRow.week, LabelRow.value).where(LabelRow.label_set == "ews_v1")
    ).all()
    labels = pd.Series({r.week: r.value for r in label_rows}, name="Y")

    df = wide.join(labels)
    df.index.name = "Data"
    df = df.reset_index()
    df["Data"] = pd.to_datetime(df["Data"])
    df["Y"] = df["Y"].astype(int)
    return df.sort_values("Data").reset_index(drop=True)
