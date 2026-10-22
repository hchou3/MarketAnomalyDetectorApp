# Symbol Coverage — Registry Update 2026-10-22

*This file documents a registry change. Measured fidelity (correlation vs EWS
reference) will populate once `mad-data backfill` and `mad-data coverage` are run
against the live database.*

---

## Architecture

Each longer-history Yahoo source is now its own top-level instrument in
`config/instruments.yaml` rather than a second source on the canonical instrument.
This avoids `(instrument_id, provider)` primary-key collisions in `symbol_map` and
`prices` (each allows at most one source per provider per instrument).

### Proxy instrument pattern

```yaml
- id: crb_commodity_bcom          # new proxy instrument
  reference_instrument: crb_commodity   # EWS column for fidelity comparison
  sources:
    - provider: yahoo
      symbol: "^BCOM"
      fidelity: proxy
```

`reference_instrument` tells the coverage report which EWS column to compare against
(e.g., `crb_commodity_bcom` is compared against CRY via `crb_commodity`).

### Splicing behaviour

To use a proxy instrument as a history extension, add its ID to
`FeatureSpec.fallback_instrument_ids`:

```python
FeatureSpec(
    col="crb_commodity_lr_1w",
    instrument_id="crb_commodity",
    fallback_instrument_ids=["crb_commodity_bcom"],
    ...
)
```

`load_prices` splices the fallback's rows into the primary's DataFrame, but only
for dates **before** the primary's first observation, and only when the fallback's
declared fidelity is `proxy` or `exact`. A `source_instrument_id` column records
which registry instrument contributed each row.

---

## New proxy instruments

### `us_hy_corp_tr_vwehx` — proxy for `us_hy_corp_tr`

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | VWEHX | yahoo | ~1978 | yes |

VWEHX (Vanguard High-Yield Corporate Bond mutual fund) is a NAV series going back
to 1978. Compared against LF98TRUU (via `reference_instrument: us_hy_corp_tr`).

### `msci_europe_iev` — proxy for `msci_europe`

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | IEV | yahoo | ~2000-07-25 | yes |

iShares Europe ETF; extends coverage from 2000 before VGK's 2005 start.
Compared against MXEU (via `reference_instrument: msci_europe`).

### `msci_europe_stoxx50` — proxy for `msci_europe`

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | ^STOXX50E | yahoo | ~1986 | no |

Euro Stoxx 50 (EUR-denominated, large-cap eurozone only); provides deep history
pre-2000 at the cost of currency mismatch and narrower universe.
Compared against MXEU (via `reference_instrument: msci_europe`).

### `msci_india_epi` — proxy for `msci_india`

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | EPI | yahoo | 2008-02-22 | yes |

WisdomTree India Earnings ETF (USD); bridges 2008–2012.
Compared against MXIN (via `reference_instrument: msci_india`).

### `msci_india_bsesn` — proxy for `msci_india`

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | ^BSESN | yahoo | ~1979 | no |

BSE Sensex (INR-denominated, 30 large-cap stocks); provides deep history
pre-2008 at the cost of currency mismatch.
Compared against MXIN (via `reference_instrument: msci_india`).

### `msci_china_fxi` — proxy for `msci_china`

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | FXI | yahoo | 2004-10-05 | yes |

iShares China Large-Cap ETF (USD); bridges 2004–2011.
Compared against MXCN (via `reference_instrument: msci_china`).

### `msci_china_sse` — proxy for `msci_china`

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | 000001.SS | yahoo | ~1990 | no |

Shanghai Composite Index (CNY-denominated); provides deep history pre-2004
at the cost of currency mismatch and different universe.
Compared against MXCN (via `reference_instrument: msci_china`).

### `crb_commodity_bcom` — proxy for `crb_commodity`

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | ^BCOM | yahoo | ~1991 | no |

Bloomberg Commodity Index (launched as DJ-AIG Commodity Index in 1998,
back-calculated to ~1991 on Yahoo). Extends coverage before DBC's 2006 start.
Compared against CRY (via `reference_instrument: crb_commodity`).

---

## Fidelity measurement checklist

Run after fetching live data:

```
mad-data backfill crb_commodity_bcom msci_europe_iev msci_europe_stoxx50
mad-data backfill msci_india_epi msci_india_bsesn msci_china_fxi msci_china_sse
mad-data backfill us_hy_corp_tr_vwehx
mad-data coverage --check
```

Expected outcomes (to confirm or override after measurement):

| Proxy instrument | Symbol | Expected suggested fidelity |
|------------------|--------|-----------------------------|
| us_hy_corp_tr_vwehx | VWEHX | proxy |
| msci_europe_iev | IEV | proxy |
| msci_europe_stoxx50 | ^STOXX50E | proxy or review_required |
| msci_india_epi | EPI | proxy |
| msci_india_bsesn | ^BSESN | proxy or review_required |
| msci_china_fxi | FXI | proxy |
| msci_china_sse | 000001.SS | proxy or review_required |
| crb_commodity_bcom | ^BCOM | proxy or review_required |

If the coverage report flags a mismatch, either add `fidelity-override:` to the
instrument's `notes` field (with justification) or change the YAML fidelity to
match the suggestion.
