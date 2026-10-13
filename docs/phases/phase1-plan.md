# Phase 1 Plan — Symbol Map and Data Layer

**Parent plan:** `architecture-plan.md`, section 6, Phase 1 (specs in 4.1, 4.2, 5.1).
**Written:** 2026-10-13, against `main` + `phase1-fixes` + `walk-forward-backtest`.
**Prerequisite:** Phase 0 merged (PR #3).
**Branches:** one branch and PR per step (`phase-1/<step>`), merged in order.

## Goal

Every instrument the app uses lives in one store, at its native frequency, with its
provenance and the time it became known. The model is not touched: `train.py` and
`backtest.py` keep reading the CSV until Phase 2 switches them to the database.

## Constraints that shape this plan

- **Claude can't reach Yahoo, FRED, or Cboe** from its workspace. Claude builds and
  tests every provider against recorded responses (fixtures). Real fetches run on
  the user's machine or in GitHub Actions, which has open internet.
- **Point-in-time correctness matters more than coverage.** The backtest is only
  honest if every value it uses was knowable on that date. `fetch_stress_data.py`
  already applies this for FRED (11-day publication lag, no back-fill before the
  first value); Phase 1 makes it a property of the data layer instead of one script.
- **Live-ready rules (plan 2.3)** apply: `subscribe()` seam on providers, a
  `PriceSink` interface, event time plus ingestion time on every row.

---

## Step 1 — Instrument registry

**Deliverable:** `config/instruments.yaml`, a schema that validates it, and tests.

Schema (Pydantic models in `src/market_anomaly_detector/registry.py`):

```yaml
- id: usd_jpy                 # canonical ID, snake_case, never reused
  name: US dollar / Japanese yen
  asset_class: fx             # fx | rate | equity_index | commodity | volatility | credit | macro_index
  groups: [safe_haven_candidate]
  units: price                # price | percent | index_points
  quote: "JPY per USD"        # human-readable direction; higher = weaker yen
  native_frequency: daily     # daily | weekly | monthly
  sources:
    - provider: fred
      symbol: DEXJPUS
      priority: 1
      fidelity: exact         # exact | proxy | unavailable
      publication_lag_days: 1
      revised: false
    - provider: yahoo
      symbol: "JPY=X"
      priority: 2
      fidelity: exact
    - provider: bloomberg_ews
      symbol: JPY
      priority: 99            # reference only; never used for live data
      fidelity: exact
  valid_from: null
  valid_to: null
  notes: ""
```

Validation rules (tests): unique IDs; every source has a known provider; priorities
unique per instrument; `unavailable` sources have no symbol; every EWS column maps
to exactly one instrument; `revised: true` requires a note on how it's handled.

**Draft mapping** (Claude fills this in; every live symbol is a *candidate* until
step 6 verifies it against the Bloomberg series; "?" means Claude is unsure what the
Bloomberg code covers and the user should confirm):

| EWS column | Canonical ID | Live candidate(s) | Expected fidelity |
|---|---|---|---|
| XAU BGNL | gold | Yahoo `GC=F` | proxy (futures, not spot) |
| ECSURPUS | us_econ_surprise | — | unavailable (proprietary) |
| BDIY | baltic_dry | — | unavailable |
| CRY | crb_commodity | Yahoo `DBC` (from 2006) | proxy |
| DXY | dollar_index | Yahoo `DX-Y.NYB` | exact |
| JPY | usd_jpy | FRED `DEXJPUS`, Yahoo `JPY=X` | exact |
| GBP | gbp_usd | FRED `DEXUSUK`, Yahoo `GBPUSD=X` | exact |
| Cl1 | wti_front | Yahoo `CL=F`; FRED `DCOILWTICO` (spot) | exact / proxy |
| VIX | vix | Cboe `VIX`, Yahoo `^VIX`, FRED `VIXCLS` | exact |
| USGG30YR | us_30y_yield | FRED `DGS30`, Yahoo `^TYX` | exact; FRED series has a 2002–2006 gap |
| GT10 | us_10y_yield | FRED `DGS10`, Yahoo `^TNX` | exact |
| USGG2YR | us_2y_yield | FRED `DGS2` | exact |
| USGG3M | us_3m_yield | FRED `DGS3MO`, Yahoo `^IRX` | exact |
| US0001M | usd_libor_1m | — (discontinued 2023); successor SOFR on FRED | unavailable; successor separate instrument |
| GTDEM2Y / 10Y / 30Y | de_2y / de_10y / de_30y_yield | 10Y: FRED monthly long-term rate | proxy (monthly) for 10Y; others unavailable |
| EONIA | eonia | — (discontinued 2022); successor €STR | unavailable; successor separate instrument |
| GTITL2YR / 10YR / 30YR | it_2y / it_10y / it_30y_yield | 10Y: FRED monthly long-term rate | proxy (monthly) for 10Y; others unavailable |
| GTJPY2YR / 10YR / 30YR | jp_2y / jp_10y / jp_30y_yield | 10Y: FRED monthly long-term rate | as above |
| GTGBP2Y / 20Y / 30Y | uk_2y / uk_20y / uk_30y_yield | — | unavailable (UK 10Y monthly exists but isn't in the EWS set) |
| LUACTRUU | us_ig_corp_tr | Yahoo `LQD` (from 2002) | proxy |
| LF98TRUU | us_hy_corp_tr | Yahoo `HYG` (from 2007) | proxy |
| EMUSTRUU | em_usd_bond_tr | Yahoo `EMB` (from 2007) | proxy |
| LUMSTRUU | us_mbs_tr | Yahoo `MBB` (from 2007) | proxy |
| LMBITR | us_muni_tr | Yahoo `MUB` (from 2007) | proxy |
| LG30TRUU ? | global_hy_tr ? | to confirm | ? |
| LP01TREU ? | pan_euro_agg_tr ? | to confirm | ? |
| LF94TRUU ? | ? | to confirm | ? |
| MXUS | msci_usa | Yahoo `^GSPC` (index) or `EUSA` (from 2010) | proxy |
| MXEU | msci_europe | Yahoo `VGK` (FTSE index, from 2005) | proxy |
| MXJP | msci_japan | Yahoo `EWJ` | proxy |
| MXBR | msci_brazil | Yahoo `EWZ` | proxy |
| MXIN | msci_india | Yahoo `INDA` (from 2012) | proxy |
| MXCN | msci_china | Yahoo `MCHI` (from 2011) | proxy |
| MXRU | msci_russia | — (removed 2022; ETF proxies delisted) | unavailable after 2022 |

Stress series from `stress_extra.csv`, all new instruments:

| Canonical ID | Sources | Notes |
|---|---|---|
| vix3m, vix9d, vvix, skew, ovx, gvz | Cboe CSV (priority 1), Yahoo `^VIX3M` etc. (priority 2) | Daily |
| nfci | FRED `NFCI` | Weekly, Friday-dated, 11-day lag, **revised** (open decision 11) |
| stlfsi | FRED `STLFSI4` | Weekly, 11-day lag, check whether revised |

Groups (initial): `stress_indicator` (vix, vix3m, vix9d, vvix, skew, ovx, gvz, nfci,
stlfsi, it_*_yield, us_hy_corp_tr), `safe_haven_candidate` (gold, usd_jpy, us_10y_yield,
us_30y_yield, de_10y_yield), `us_rates`, `global_rates`, `fx`, `equity_index`,
`credit`, `commodity`. Groups are descriptive and editable; nothing is labeled a haven
on assertion alone (plan 4.1).

**Done when:** registry validates in CI, all 42 EWS columns and 8 stress series are
mapped, and the "?" rows are confirmed or left `unavailable` with a note.

---

## Step 2 — Database

**Deliverable:** Postgres in Docker Compose, SQLAlchemy 2 models, the first Alembic
migration, and a small repository layer.

Tables:

| Table | Key columns | Notes |
|---|---|---|
| `instruments` | `id` PK, name, asset_class, units, quote, native_frequency, valid_from/to | Synced from the YAML |
| `instrument_groups` | (`instrument_id`, `group`) | Many-to-many |
| `symbol_map` | (`instrument_id`, `provider`) PK, symbol, priority, fidelity, **source_frequency**, publication_lag_days, revised | Synced from the YAML |
| `prices` | (`instrument_id`, `provider`, `obs_date`) PK, value, `known_at`, `ingested_at`, `run_id` | Raw, native frequency. A `vintage` column is added only if open decision 11 picks ALFRED |
| `labels` | (`label_set`, `week`) PK, value, source | `ews_v1` from the CSV |
| `ingestion_runs` | id, started/finished, mode, status, counts | One row per CLI run |
| `ingestion_run_items` | (`run_id`, `instrument_id`, `provider`), status, rows, first/last date, error | Per-instrument outcome |

Rules: `mad-data sync-registry` upserts instruments and symbol map from YAML (YAML is
the source of truth); removing an instrument from YAML marks it inactive, never
deletes its prices. Upserts are idempotent: re-running an ingest changes nothing
unless the provider changed a value, and a changed value is logged.

**`known_at` computation:** `known_at = obs_period_end(obs_date, source_frequency) +
publication_lag_days`. For `source_frequency = monthly`, `obs_period_end` returns the
last calendar day of the observation month, because FRED dates monthly values on the
1st of the month (e.g. `2023-01-01` represents January 2023, which ends on the 31st).
For daily and weekly sources `obs_period_end` is the identity. This means a monthly
FRED series with `publication_lag_days = 40` and `obs_date = 2023-01-01` yields
`known_at = 2023-01-31 + 40 days = 2023-03-12`. Never use the raw `obs_date` as the
period start when computing `known_at` for monthly series.

Tests run against real Postgres (Docker Compose locally, a service container in CI).
No SQLite substitute, since upsert syntax and timestamp handling differ.

**Done when:** `docker compose up db && alembic upgrade head` works on a clean
machine, and repository tests pass in CI.

---

## Step 3 — Load the EWS CSV

**Deliverable:** `EwsCsvProvider` and `mad-data load-ews`.

- Each of the 42 series goes into `prices` with `provider = bloomberg_ews` and
  `known_at = obs_date` (the CSV's own convention: values as of that Tuesday).
- `Y` goes into `labels` as label set `ews_v1`.
- Parity test: rebuilding the wide weekly frame from the database reproduces the CSV
  exactly (same dates, columns, and values to the file's 3 decimals). This is what lets
  Phase 2 switch `train.py`/`backtest.py` to the database with no metric change.

**Done when:** parity test passes and the backtest run from a database-built frame
matches the CSV-built run exactly (script, not yet wired into `backtest.py`).

---

## Step 4 — Live providers

**Deliverable:** `YahooProvider`, `FredProvider`, `CboeProvider` behind one interface,
with recorded fixtures.

```python
class Provider(Protocol):
    name: str
    def get_history(self, symbol: str, start: date | None, end: date | None) -> pd.DataFrame: ...
        # columns: obs_date (date), value (float); sorted, no duplicates
    def subscribe(self, symbols: list[str]) -> Iterator[Tick]:  # Phase 8
        raise NotImplementedError
```

- **Yahoo** (`yfinance`): explicit `auto_adjust` (adjusted for ETFs, raw for indices and
  futures, set per source in YAML); strip timezone and normalize to the exchange date
  (as `fetch_stress_data.py` does); empty frame is an error, not success.
- **FRED:** the public `fredgraph.csv` endpoint (no key) by default; a `FRED_API_KEY`
  env var enables the official API, needed for vintages (open decision 11). Missing
  values (`.`) become NaN and are dropped.
- **Cboe:** port the two CDN URL templates and column handling from `fetch_stress_data.py`.
- **Fallback:** an instrument tries sources in priority order; the row records which
  provider served it. A fallback is logged as a warning in the run summary.
- **Network policy:** timeouts, retries with backoff on 5xx and timeouts only,
  a polite delay between Yahoo requests, `certifi` CA bundle (the Windows SSL fix from
  the existing script).
- **Fixtures:** one recorded response per provider per scenario (normal, empty,
  malformed, HTTP error) under `tests/fixtures/`. A `scripts/record_fixtures.py`
  refreshes them; the user runs it, since Claude can't reach the sites. Tests never
  hit the network unless marked `@pytest.mark.network`; a weekly CI job runs those.

**Verify during Step 4:** FRED series DEXJPUS (`usd_jpy`) and DEXUSUK (`gbp_usd`)
are described as "daily" but have historically been published only on business days
with some gaps. Confirm whether they carry a 5-day or 7-day cadence in practice and
whether the H.10 `publication_lag_days = 3` is the right floor, or if a higher value
is needed when a holiday falls on the Monday after a Friday `obs_date`.

**Done when:** provider tests pass offline, and the user has run the network tests
once successfully.

---

## Step 5 — Ingestion CLI

**Deliverable:** `mad-data` command group.

| Command | Does |
|---|---|
| `mad-data sync-registry` | YAML → `instruments`, `symbol_map` |
| `mad-data load-ews` | Step 3 |
| `mad-data backfill [--instrument ID ...] [--since DATE]` | Full history for each active instrument via its best available source |
| `mad-data update` | Since last stored date minus 5 days (catches revisions), all active instruments |
| `mad-data status` | Last run, stale instruments, per-instrument coverage |

Behavior:
- One instrument failing never fails the run; the exit code is non-zero only if every
  instrument failed or the database is unreachable.
- `known_at` = `obs_date` + the source's `publication_lag_days` (0 for daily market
  closes). Stored, never recomputed at read time.
- Validation per batch before writing (plan 5.1): duplicate dates, non-numeric
  values, out-of-range jumps flagged (not dropped) for review, negative values only
  where allowed (WTI, yields), series shorter than the longest planned feature window
  flagged.
- Staleness: an instrument whose last `obs_date` is older than its frequency allows
  (e.g. 5 trading days for daily) is marked stale in `mad-data status`.
- Writes go through a `PriceSink` (Postgres now, Kafka in Phase 8).
- No scheduler yet. The daily job is a CLI call; the CronJob comes in Phase 5.

**Done when:** backfill then update is idempotent (second run writes 0 changed rows),
and a simulated provider failure is isolated to its instrument.

---

## Step 6 — Coverage and fidelity report

**Deliverable:** `mad-data report` → `docs/reports/symbol-coverage.md` (regenerated,
committed) plus a CSV with the same numbers.

Per instrument: chosen source, first/last date, row count, gaps longer than 5 trading
days, staleness, and, where a Bloomberg EWS series exists, a comparison over the
overlap on Tuesday-aligned values:
- correlation of weekly changes (the number that matters for features)
- median ratio of levels (catches unit/scale mistakes, e.g. percent vs. basis points)
- sign check (catches an inverted quote direction)

Suggested fidelity thresholds, recorded in the report and then set in YAML by hand:
weekly-change correlation ≥ 0.98 → `exact`, ≥ 0.80 → `proxy`, below → flag for review.

**Done when:** the user has run `backfill` + `report` once with network access, and
the YAML fidelity values match the report.

---

## Step 7 — Retire the stress script, update docs

- Parity test: database values for the 8 stress series, aligned to EWS Tuesdays with
  the same lag rule, match `data/raw/stress_extra.csv` on the overlap (tolerance for
  vendor revisions, which the test reports rather than hides).
- Delete `scripts/fetch_stress_data.py` and `stress_extra.csv` once parity holds, or
  keep the CSV as a frozen snapshot if the user prefers (D6).
- Update `data/README.md`, `README.md`, and plan section 3; log decisions in section 8.

---

## Acceptance checks for Phase 1

- [ ] Registry validates; every EWS column and stress series is mapped
- [ ] `docker compose up db`, `alembic upgrade head`, `mad-data sync-registry`,
      `mad-data load-ews` work on a clean machine
- [ ] EWS parity: database frame equals the CSV exactly
- [ ] Offline test suite passes in CI; network tests pass when the user runs them
- [ ] `backfill` then `update` is idempotent
- [ ] Coverage report committed; YAML fidelity values agree with it
- [ ] Stress-series parity holds or differences are explained in the report
- [ ] `train.py` and `backtest.py` results unchanged from the Phase 0 baseline
- [ ] `git status` clean after running everything

---

## Decisions (defaults unless the user says otherwise)

| # | Decision | Default |
|---|---|---|
| D1 | Postgres from Phase 1, or a lighter store (SQLite/DuckDB) first? | **Decided 2026-10-13: Postgres** in Docker Compose |
| D2 | Store raw or transformed values | Raw; transforms at feature time |
| D3 | Revised FRED series (open decision 11) | Ingest current history now, mark `revised: true`; Phase 2 doesn't use them as features until decided |
| D4 | Primary source when FRED and Yahoo both cover an instrument | FRED for rates and FX (official, long history); Yahoo for indices, futures, ETFs |
| D5 | Unclear Bloomberg codes (LG30TRUU, LP01TREU, LF94TRUU) | **Decided 2026-10-13:** `unavailable`, with a note that the index is unidentified. They stay in the EWS data as `bloomberg_ews` series |
| D6 | Keep `stress_extra.csv` after parity? | Keep as a frozen snapshot in `data/raw/` with its fetch date; delete the script |
| D7 | Labels for 2021-05 onward (open decision 1) | Not decided in Phase 1; the `labels` table supports adding a new label set later |

## Out of scope

Feature engineering, model changes, the scheduler/CronJob, API endpoints, news, and
any switching of `train.py`/`backtest.py` to the database (Phase 2).
