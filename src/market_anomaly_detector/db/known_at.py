"""Point-in-time helpers: compute when an observation first became known.

FRED dates monthly observations on the first of the month (e.g. 2023-01-01 means
"January 2023").  The observation period ends on 2023-01-31, so known_at must be
computed from the *end* of the period, not the obs_date, to avoid leaking future data
into backtests.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta


def obs_period_end(obs_date: date, source_frequency: str | None) -> date:
    """Return the last calendar day of the observation period that obs_date belongs to.

    For daily and weekly sources the obs_date already marks the end of its period.
    For monthly sources the period runs to the last day of the month.
    """
    if source_frequency == "monthly":
        last_day = calendar.monthrange(obs_date.year, obs_date.month)[1]
        return obs_date.replace(day=last_day)
    return obs_date


def compute_known_at(
    obs_date: date,
    source_frequency: str | None,
    publication_lag_days: int,
) -> date:
    """Return the earliest date on which this observation was publicly available.

    known_at = obs_period_end(obs_date, source_frequency) + publication_lag_days
    """
    return obs_period_end(obs_date, source_frequency) + timedelta(days=publication_lag_days)
