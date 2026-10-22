# FX / Equity Timing Diagnostic

*Stub report — run `mad-data timing-diagnostic` against the live database to populate.*

This report computes weekly-change correlation between a set of instruments and their
bloomberg_ews reference series at obs_date shifts of −1, 0, and +1 business day.

## Instruments Under Investigation

| Instrument | Provider | Symbol | Hypothesis |
|---|---|---|---|
| gbp_usd | yahoo | GBPUSD=X | WM/Reuters 4 pm fix may lag EWS snapshot by 1 bd |
| usd_jpy | yahoo | JPY=X | WM/Reuters 4 pm fix may lag EWS snapshot by 1 bd |
| gbp_usd_fred | fred | DEXUSUK | FRED DEXUSUK uses Fed noon rate; timing reference |
| msci_japan | yahoo | EWJ | US-hours close vs EWS MXJP Wednesday snapshot |
| msci_japan_n225 | yahoo | ^N225 | Tokyo-hours close; may align better at 0 or −1 bd |

## Correlation at obs_date Shifts

*(Run `mad-data timing-diagnostic` to fill this table.)*

| Instrument | Provider | Symbol | EWS Reference | corr(−1 bd) | corr(0) | corr(+1 bd) | Best Shift | Best Corr | Note |
|---|---|---|---|---|---|---|---|---|---|
| gbp_usd | yahoo | GBPUSD=X | gbp_usd | — | 0.889 | — | 0 | — | baseline from coverage report |
| usd_jpy | yahoo | JPY=X | usd_jpy | — | 0.883 | — | 0 | — | baseline from coverage report |
| gbp_usd_fred | fred | DEXUSUK | gbp_usd | — | 0.989 | — | 0 | — | baseline from coverage report |
| msci_japan | yahoo | EWJ | msci_japan | — | 0.787 | — | 0 | — | baseline from coverage report |
| msci_japan_n225 | yahoo | ^N225 | msci_japan | — | — | — | 0 | — | not yet ingested |

## Interpretation

- **corr ≥ 0.98** at a non-zero shift → candidate for an obs_date/known_at correction rule.
- **corr ≥ 0.80** → proxy-grade; timing adjustment optional.
- **corr < 0.80** → below proxy threshold; consider replacing the source.

## Proposed Correction Rules

*(Populate after running `mad-data timing-diagnostic` against the live database.)*

No correction rules applied yet — instrument YAML obs_date fields are unchanged.

## Raw Data Note

This report is generated from the local fixture database only; live data was not
fetched during generation.  Re-run after `mad-data backfill` to refresh.
