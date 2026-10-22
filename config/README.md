# Instrument Registry

`instruments.yaml` is the single source of truth for every instrument the
pipeline knows about.  Each entry is validated on load by Pydantic (see
`src/market_anomaly_detector/registry.py`).

## Key fields

| Field | Description |
|---|---|
| `id` | Unique snake_case identifier used throughout the codebase |
| `active` | Set `false` to exclude from backfill and staleness checks |
| `native_frequency` | Underlying data cadence (`daily`, `weekly`, `monthly`) |
| `sources[].publication_lag_days` | Calendar days between `obs_date` and public availability; sets `known_at` to prevent look-ahead bias |
| `sources[].source_frequency` | Override when a source publishes at a coarser cadence than the instrument's `native_frequency` (e.g. OECD monthly yields for a daily-frequency instrument) |

## Incomplete-bar rule

Daily bars dated **today** (exchange local time) are **never stored** by
`run_backfill`.  A bar is only written on the next run, after the session has
closed.

Rationale: intraday fetches from Yahoo or Cboe return a partial bar whose
OHLCV values change as the day progresses.  Storing a 1 pm snapshot as the
official close would corrupt the `known_at` timeline.

Implementation: `run_backfill` passes `cutoff_date=date.today()` to
`_process_one`, which filters out any `obs_date >= cutoff_date` rows from
daily sources before writing.  Weekly and monthly sources are not filtered.

## Staleness thresholds

Staleness is checked against the **priority-1 source** only (lower-priority
fallbacks are fetched only when the primary fails and are not independently
stale).  The effective grace period is `threshold + publication_lag_days`:

| `source_frequency` | Threshold | Example (lag=40) |
|---|---|---|
| `daily` | 5 business days | 5 bd + 40 cd ≈ 48 cd |
| `weekly` | 10 calendar days | 10 + 40 = 50 cd |
| `monthly` | 75 calendar days | 75 + 40 = 115 cd |

OECD monthly yields (`de/it/jp_10y_yield`) use `publication_lag_days: 40`,
giving ~115 calendar days of total grace before a stale flag appears.

## Validator constraints

- Monthly sources must have `publication_lag_days >= 28` (enforced by
  `monthly_source_needs_realistic_lag`).
- `publication_lag_days` must reflect the **real publication delay**, because
  it is used to set `known_at`.  Do not inflate it to work around staleness
  flags; fix the staleness threshold instead.
