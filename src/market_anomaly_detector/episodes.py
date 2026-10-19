"""Episode extraction for the Market Anomaly Detector.

An episode is a contiguous run of label-1 weeks.  Short calm gaps (up to
*merge_gap* consecutive calm weeks) are bridged so nearby stress events form
one episode.  Shared with Phase 3 (per-instrument episode stats).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Episode:
    """A contiguous anomaly episode (possibly spanning bridged calm gaps).

    Attributes
    ----------
    start : date
        First anomaly week of the episode (onset).
    end : date
        Last anomaly week of the episode.
    peak_week : date
        Set to *start* (onset week) when constructed from binary labels alone.
        Phase 3 enriches this with the week of maximum price-based stress.
    length : int
        Total weeks spanned from *start* to *end* inclusive, counting any
        bridged calm weeks.
    """

    start: date
    end: date
    peak_week: date
    length: int


def extract_episodes(
    labels: pd.DataFrame,
    merge_gap: int = 2,
) -> list[Episode]:
    """Extract anomaly episodes from a sorted labels DataFrame.

    Parameters
    ----------
    labels:
        DataFrame with ``week`` (date) and ``value`` (int 0/1), sorted by
        week ascending.  Weeks are assumed to be spaced ~7 days apart.
    merge_gap:
        Maximum number of consecutive calm weeks between two anomaly runs that
        are merged into one episode.  Runs separated by more than *merge_gap*
        calm weeks remain separate episodes.

    Returns
    -------
    list[Episode]
        Episodes in chronological order.  Empty list if there are no anomaly
        weeks or *labels* is empty.
    """
    if labels.empty or not (labels["value"] == 1).any():
        return []

    df = labels.copy().reset_index(drop=True)
    weeks: list[date] = df["week"].tolist()
    values: list[int] = df["value"].astype(int).tolist()
    n = len(weeks)

    # Collect raw contiguous runs of label=1: (start_idx, end_idx) inclusive.
    raw_runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if values[i] == 1:
            j = i + 1
            while j < n and values[j] == 1:
                j += 1
            raw_runs.append((i, j - 1))
            i = j
        else:
            i += 1

    if not raw_runs:
        return []

    # Merge runs whose calm gap (in calendar weeks) is <= merge_gap.
    # Gap is measured between the last week of the previous run and the first
    # week of the next run: gap_weeks = round-trip calendar weeks - 1.
    merged: list[list[int]] = [list(raw_runs[0])]
    for start_idx, end_idx in raw_runs[1:]:
        prev_end_idx = merged[-1][1]
        gap_days = (weeks[start_idx] - weeks[prev_end_idx]).days
        gap_weeks = gap_days // 7 - 1
        if gap_weeks <= merge_gap:
            merged[-1][1] = end_idx
        else:
            merged.append([start_idx, end_idx])

    episodes: list[Episode] = []
    for start_idx, end_idx in merged:
        start = weeks[start_idx]
        end = weeks[end_idx]
        length = (end - start).days // 7 + 1
        episodes.append(Episode(start=start, end=end, peak_week=start, length=length))

    return episodes
