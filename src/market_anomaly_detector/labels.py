"""Label loading and target construction.

Labels live in the ``labels`` table under a named label set (default ``ews_v1``).

Functions
---------
load_labels(session, label_set) -> pd.DataFrame
    Pull week/value rows from the database for one label set.

build_targets(labels, horizon) -> pd.DataFrame
    Add ``detect`` (label at week t) and ``predict`` (any label in
    weeks t+1..t+horizon) columns.  The last *horizon* rows receive
    ``predict = NaN`` because there is not enough future data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

DEFAULT_LABEL_SET = "ews_v1"
DEFAULT_HORIZON = 4


def load_labels(
    session: Session,
    label_set: str = DEFAULT_LABEL_SET,
) -> pd.DataFrame:
    """Load labels from the database for *label_set*.

    Returns
    -------
    pd.DataFrame
        Columns: ``week`` (date), ``value`` (int 0/1), sorted by week ascending.
        Empty DataFrame (same columns) if no labels are found.
    """
    from sqlalchemy import select

    from .db.models import LabelRow

    rows = session.execute(
        select(LabelRow.week, LabelRow.value)
        .where(LabelRow.label_set == label_set)
        .order_by(LabelRow.week)
    ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["week", "value"])

    df = pd.DataFrame(rows, columns=["week", "value"])
    df["week"] = pd.to_datetime(df["week"]).dt.date
    df["value"] = df["value"].astype(int)
    return df.sort_values("week").reset_index(drop=True)


def build_targets(
    labels: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Build detect and predict targets from a labels DataFrame.

    Parameters
    ----------
    labels:
        DataFrame with columns ``week`` (date) and ``value`` (int 0/1),
        sorted by week ascending.
    horizon:
        Predict window in weeks.  At week index *i*, ``predict[i]`` is 1 if
        any of weeks *i+1* through *i+horizon* has ``value = 1``, else 0.
        The last *horizon* rows receive ``predict = NaN`` because fewer than
        *horizon* future weeks exist.

    Returns
    -------
    pd.DataFrame
        Columns: ``week``, ``detect`` (float 0/1), ``predict`` (float 0/1/NaN).
    """
    if labels.empty:
        return pd.DataFrame(columns=["week", "detect", "predict"])

    df = labels[["week", "value"]].copy().reset_index(drop=True)
    n = len(df)
    values = df["value"].values.astype(int)

    predict = np.full(n, np.nan)
    for i in range(n - horizon):
        predict[i] = float(int(np.any(values[i + 1 : i + horizon + 1])))

    return pd.DataFrame(
        {
            "week": df["week"],
            "detect": df["value"].astype(float),
            "predict": predict,
        }
    )
