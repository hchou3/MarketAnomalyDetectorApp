# Symbol Coverage — Registry Update 2026-10-22

*This file documents a registry change. Measured fidelity (correlation vs EWS
reference) will populate once `mad-data backfill` and `mad-data coverage` are run
against the live database.*

---

## What changed

Longer-history Yahoo proxies added for five instruments. Each new source is
declared `fidelity: proxy`; the coverage report will confirm or flag that
declaration when live data is available. The existing primary sources (lowest
priority number) are unchanged.

### Splicing behaviour

`load_prices` now splices earlier-starting secondaries onto the primary when
the secondary's declared fidelity is `proxy` or `exact`. The splice is
backward-only: secondary rows are used only for dates **before** the primary's
first observation. A `source_provider` column on the returned DataFrame records
which provider contributed each row.

---

## New sources

### `us_hy_corp_tr` — Bloomberg US HY Corporate Bond TR

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | HYG | yahoo | 2007-04-11 | yes |
| **2** | **VWEHX** | **yahoo** | **~1978** | **yes** |
| 99 | LF98TRUU | bloomberg_ews | — | — |

VWEHX (Vanguard High-Yield Corporate Bond mutual fund) is a mutual-fund NAV
series going back to 1978. It fills the pre-2007 window where HYG has no data.
Both track the broad US HY market; NAV and ETF pricing should correlate strongly
during overlap.

### `msci_europe` — MSCI Europe Index

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | VGK | yahoo | 2005-03-04 | yes |
| **2** | **IEV** | **yahoo** | **~2000-07-25** | **yes** |
| **3** | **^STOXX50E** | **yahoo** | **~1986** | **no** |
| 99 | MXEU | bloomberg_ews | — | — |

IEV (iShares Europe ETF) extends coverage to 2000. ^STOXX50E (Euro Stoxx 50) is
EUR-denominated and covers only large-cap eurozone names — expect lower
correlation with MSCI Europe (USD, broad) but usable as a proxy pre-2000.

### `msci_india` — MSCI India Index

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | INDA | yahoo | 2012-02-09 | yes |
| **2** | **EPI** | **yahoo** | **2008-02-22** | **yes** |
| **3** | **^BSESN** | **yahoo** | **~1979** | **no** |
| 99 | MXIN | bloomberg_ews | — | — |

EPI (WisdomTree India Earnings ETF, USD) bridges 2008–2012. ^BSESN (BSE
Sensex, INR) provides deep history but carries a currency mismatch; fidelity
vs the USD-denominated MSCI India may be "review_required" — let the coverage
report decide.

### `msci_china` — MSCI China Index

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | MCHI | yahoo | 2011-03-29 | yes |
| **2** | **FXI** | **yahoo** | **2004-10-05** | **yes** |
| **3** | **000001.SS** | **yahoo** | **~1990** | **no** |
| 99 | MXCN | bloomberg_ews | — | — |

FXI (iShares China Large-Cap ETF, USD) bridges 2004–2011. 000001.SS (Shanghai
Composite, CNY, A-shares) provides deep history; currency mismatch and different
universe — coverage report will measure actual correlation.

### `crb_commodity` — Thomson Reuters/CoreCommodity CRB Index

| Priority | Symbol | Provider | From | Auto-adjust |
|----------|--------|----------|------|-------------|
| 1 | DBC | yahoo | 2006-02-03 | yes |
| **2** | **^BCOM** | **yahoo** | **~1991** | **no** |
| 99 | CRY | bloomberg_ews | — | — |

^BCOM (Bloomberg Commodity Index, launched as DJ-AIG Commodity Index in 1998,
back-calculated to ~1991 on Yahoo) is the longest public commodity index
available on Yahoo with pre-2006 coverage. It tracks a different basket from CRY
(Reuters/Jefferies CRB); expect `proxy` fidelity at best. Verify the actual
Yahoo start date by running `mad-data backfill crb_commodity`.

---

## Fidelity measurement checklist

Run after fetching live data:

```
mad-data backfill us_hy_corp_tr msci_europe msci_india msci_china crb_commodity
mad-data coverage --check
```

Expected outcomes (to confirm or override after measurement):

| Instrument | New source | Expected suggested fidelity |
|------------|------------|----------------------------|
| us_hy_corp_tr | VWEHX | proxy |
| msci_europe | IEV | proxy |
| msci_europe | ^STOXX50E | proxy or review_required |
| msci_india | EPI | proxy |
| msci_india | ^BSESN | proxy or review_required |
| msci_china | FXI | proxy |
| msci_china | 000001.SS | proxy or review_required |
| crb_commodity | ^BCOM | proxy or review_required |

If the coverage report flags a mismatch, either add `fidelity-override:` to the
instrument's `notes` field (with justification) or change the YAML fidelity to
match the suggestion.
