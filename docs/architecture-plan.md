# Market Anomaly Detector — Architecture Plan & Build Instructions

**Status:** Active plan as of 2026-10-13. Phase 0 is complete on
`phase-0/repo-hygiene-v2` (pending PR review); Phase 1 is next. This plan reorders
the original spec (`docs/archive/market-anomaly-detector-project.md`) into four tiers:
1. **Batch core** (Phases 0–4): data, model, episodes, API.
2. **Required serving architecture** (Phase 5): Triton behind a gRPC scoring service,
   a Redis cache, deployed on Kubernetes. These are project requirements and do not
   depend on live data, so they come before the live system.
3. **Product layer** (Phases 6–7): narrator, news, frontend.
4. **Live system** (Phase 8): Kafka, Flink or PySpark, Feast, FAISS, and streaming
   on top of the Phase 5 serving stack.

The modeling goals from `docs/archive/phase1-build-instructions.md` still apply except
where noted here.

This file is the working brief for Claude when building in this repo. Read it in full
before starting a task, follow the phase order, and update the "Current state" and
"Decision log" sections when something changes. Phase plans live in `docs/phases/`;
measured results in `docs/reports/`.

---

## 1. Product scope

**What the app does:** detects market-wide stress regimes ("anomalies") from a tracked
set of macro instruments, explains why, and shows how those instruments behaved during
past anomaly episodes.

**In scope**
- Market-level regime detection with a calibrated probability and an explanation.
- A history of anomaly episodes and how tracked instruments behaved in each one.
- A user-editable set of tracked instruments, organized into groups.
- CSV upload as an alternative data source.
- Model serving through Triton behind a gRPC scoring service, a Redis cache for
  frequently queried results, and deployment on Kubernetes (Phase 5, required).
- A live mode (Phase 8) that ingests real-time ticks and scores continuously, built on
  Kafka, Flink or PySpark, and Feast, plus FAISS similar-episode search.
- News headlines around detected anomalies and past episodes (context only).
- Plain-language narration of computed results.

**Out of scope (decided 2026-10-13)**
- Buy/sell/hold recommendations, "safe to buy" verdicts, or trading strategies.
- Ticker suggestions, peer recommendations, or valuation ("over-inflated") analysis.
- Per-ticker anomaly scores. The model is market-level; individual tickers are only
  described by how they behaved during market-wide episodes.

**Language rule:** all user-facing text describes history and model output
("gold rose in 4 of 6 past episodes"), never advice ("gold is safe to buy").

---

## 2. Target architecture

### 2.1 After Phase 5 (batch data, required serving stack)

```
 Data providers         Ingestion (K8s CronJob)          Storage
 ┌──────────────┐ fetch ┌──────────────────┐  upsert  ┌──────────────────────┐
 │ yfinance     │──────►│ daily, uses      │────────►│ Postgres             │
 │ FRED, Cboe   │       │ symbol map,      │         │  instruments         │
 │ news API     │       │ validates + logs │         │  symbol_map, prices  │
 │ CSV upload   │──────►│ (same validator) │         │  anomaly_episodes    │
 └──────────────┘       └──────────────────┘         │  regime_scores       │
                                                      │  model_versions      │
                                                      │  news_items, uploads │
                                                      └──────────┬───────────┘
                                                                 │
   ┌──────────┐  REST  ┌───────────┐  gRPC  ┌──────────────────┐ │
   │ Frontend │◄──────►│  FastAPI  │◄──────►│ Scoring service  │◄┘
   └──────────┘        │  gateway  │        │ features, SHAP,  │
                       └─────┬─────┘        │ cache logic      │
                             │              └───┬─────────┬────┘
                             │                  │ gRPC    │
                             │           ┌──────┴─────┐ ┌─┴──────┐
                             │           │  Triton    │ │ Redis  │
                             │           │ ONNX models│ │ cache  │
                             │           └────────────┘ └────────┘
              ┌──────────────┼─────────────────┐
     ┌────────┴───────┐ ┌────┴───────────┐ ┌───┴────────────┐
     │Episode analyzer│ │ News service   │ │ LLM narrator   │
     └────────────────┘ └────────────────┘ └────────────────┘

        Every box above runs as its own Kubernetes workload.
```

### 2.2 Phase 8 additions (live)

```
 Real-time provider ─► Kafka (ticks) ─► Flink/PySpark ─► Feast online store
 Replay producer ────┘                   (shared feature spec)     │
                                                                   ▼
                                       Scoring service ─► Triton, Redis
                                              │
                                    Kafka (scores) ─► FastAPI WebSocket
 FAISS index of historical anomaly weeks ─► /v1/similar
```

**Stack:** Python 3.12–3.13, pandas, scikit-learn, SHAP, skl2onnx + onnxruntime,
NVIDIA Triton Inference Server (ONNX Runtime backend), grpcio + protobuf,
FastAPI, Redis, SQLAlchemy 2 + Alembic, Postgres (plain Postgres first; TimescaleDB
only if price queries get slow), pytest, ruff, uv. Docker Compose for fast local
iteration; Kubernetes (`kind` or `minikube`) as the deployment target from Phase 5.
Frontend stack is an open decision (section 9).

### 2.3 Live-ready rules (apply from Phase 0)
These keep earlier phases compatible with the serving and live stacks:
- **Providers have a streaming seam.** The provider interface defines
  `get_history(...)` now and reserves `subscribe(symbols) -> iterator[Tick]` for
  Phase 8. Ingestion writes through one `PriceSink` interface (Postgres now, a Kafka
  producer later).
- **Features are pure, windowed functions.** `features.build()` takes a time-ordered
  frame and depends only on trailing windows, with no global state, so the same
  definitions can be ported to Flink/PySpark and registered in Feast. Keep a
  feature spec (name, inputs, window, transform) in one module that both paths read.
- **Event time everywhere.** Every price, feature, and score row carries the event
  timestamp it describes plus an ingestion timestamp. Never use wall-clock "now"
  inside feature or scoring code.
- **Scoring is a service boundary.** The API calls a `Scorer` interface
  (`score(request) -> ScoreResult`), never the model object. Phase 4 implements it
  in-process; Phase 5 replaces it with the gRPC client to the scoring service.
- **Models must be ONNX-exportable.** Prefer estimators `skl2onnx` supports. Each
  artifact build exports to ONNX and fails CI if ONNX predictions differ from the
  sklearn model beyond tolerance. Check this from Phase 2, not Phase 5.
- **One container per service** (api, scorer, scheduler, and later triton), so the
  Kubernetes manifests map one-to-one.
- **Replayable history.** Ingestion can replay stored prices in event-time order at
  a compressed clock. This becomes the Kafka replay producer for backtesting.

---

## 3. Current state (after Phase 0, 2026-10-13)

**Branches.** `main` holds the merged `phase1-fixes` and `walk-forward-backtest` work.
`phase-0/repo-hygiene-v2` holds Phase 0 on top of it, pending review. The first
attempt, `phase-0/repo-hygiene`, was built from the old `main` and is superseded.

**Headline results** (walk-forward backtest, 14 yearly folds, 2008 to 2021-04, 695
out-of-sample weeks, 153 anomalies; full report in
`docs/reports/2026-10-10-walk-forward-backtest.md`):

| Model | Precision | Recall | F1 | False-alarm rate |
|---|---|---|---|---|
| Ensemble | 0.72 | 0.56 | 0.63 | 6% |
| Baseline IsolationForest | 0.32 | 0.75 | 0.45 | 44% |
| Always flag anomaly | 0.22 | 1.00 | 0.36 | 100% |

Ensemble ROC AUC 0.88 / PR AUC 0.71 (baseline 0.77 / 0.62; prevalence 0.22).
Year-level bootstrap: F1 difference 95% CI [−0.10, +0.41], P(ensemble better) = 0.92,
so the improvement is likely but not established. Known failures: **2008 missed
entirely (0 of 27 anomaly weeks)** and **22 false alarms in 2020**, mostly in
calm-labelled weeks between and after the 2020 anomaly weeks.

| Area | State | Action |
|---|---|---|
| Layout | `src/market_anomaly_detector/` (data, splits, pipeline, explain, train, backtest, paths); top-level `tests/`; `docs/`; `data/raw/`; `scripts/` | Done (Phase 0) |
| Tooling | `pyproject.toml`; `uv.lock` pinned to `requirements.txt` (numpy 2.2.6, scipy 1.15.3, pandas 2.3.1, scikit-learn 1.6.1, shap 0.50.0, joblib 1.5.1); Python 3.12–3.13; ruff; CI on 3.12 and 3.13; commands `mad-train`, `mad-backtest` | Done (Phase 0). Results depend on these versions |
| Tests | 32 passing on 3.12 and 3.13 | Keep green |
| `data.py` | Features are raw price **levels** + MA crosses + `USD_Over_100` | Stationary features (Phase 2) |
| `splits.py` / `train.py` | Single 9y/2y/1y split; test year is 60% positive. `train.py` writes a `.meta.json` sidecar (versions, splits, calibration report, metrics) | Headline metrics come from the backtest, not this split |
| `backtest.py` | Expanding-window yearly folds 2008–2021; val window extended back until ≥ 8 anomalies; pooled metrics, trivial baselines, AUCs, bootstrap | Extend in Phase 2: event-level metrics, predict head, new features |
| `pipeline.py` | IsoForest score → StackingClassifier → calibration (`FrozenEstimator`), method chosen by OOF Brier on the val split | Re-evaluate after features change; selection by Brier can pick isotonic with a much worse log loss; verify ONNX export |
| `explain.py` | SHAP on the RF base learner only (class-1 extraction fixed and tested) | Explain the final model output (Phase 2) |
| `scripts/fetch_stress_data.py` | Cboe/Yahoo/FRED stress series aligned to EWS Tuesdays with a FRED publication lag; output `data/raw/stress_extra.csv` | Replaced by Phase 1 providers |
| Data | 42 Bloomberg-coded series; ~1/3 have no Yahoo/FRED source; labels end 2021-04; CSV source and license unknown in a public repo | Symbol map (Phase 1); open decisions 1–3 |
| Network | Claude's workspace cannot reach Yahoo, FRED, or Cboe (egress policy) | Live fetches run on the user's machine or in GitHub Actions; Claude builds against recorded fixtures |
| Lint exemptions | E501 off for `backtest.py` (report f-strings) and `scripts/fetch_stress_data.py` | Remove when those files are reworked |

---

## 4. Component specs

### 4.1 Symbol map and instrument groups
The model, database, and API refer to instruments by **canonical ID** only
(`gold`, `usd_jpy`, `vix`, `us_10y_yield`). Vendor symbols live in one config file,
`config/instruments.yaml`, loaded into the `instruments` and `symbol_map` tables.

Each instrument entry has:
- `id`, `name`, `asset_class` (fx, rate, equity_index, commodity, volatility, credit)
- `groups`: list, e.g. `[safe_haven_candidate]`, `[stress_indicator]`, `[us_rates]`
- `sources`: per-provider symbol, e.g. `{yahoo: "GC=F", fred: null, bloomberg: "XAU BGNL"}`
- `fidelity`: `exact` | `proxy` | `unavailable` per source, with a note for proxies
- `units` and `transform`: percent vs. price, quote direction (USD/JPY vs. JPY/USD), scale
- `valid_from` / `valid_to`: covers discontinued series (EONIA, USD LIBOR, MSCI Russia)
  and renames

Rules:
- Indicators and havens are separate groups. An instrument used as a stress signal
  (e.g. Italian yields) is never labeled a haven.
- Haven membership is descriptive and backed by per-episode history (4.4), not asserted.
- Users can add instruments and groups; user-added instruments are not model features
  unless the model is retrained to include them.
- Ship `scripts/check_symbols.py`: fetches each mapped symbol, reports first/last date,
  row count, and gaps. Run it in CI weekly and before any training run.
- Each source entry has a `priority` (fallback order, e.g. Cboe then Yahoo) and an
  availability rule: `publication_lag` (e.g. FRED weekly indices dated Friday are
  usable 11 days later) and `revised: true|false` for series whose history is
  revised after release (e.g. NFCI).
- The Bloomberg EWS columns are loaded as their own source (`bloomberg_ews`), so
  proxies can be compared against the original series over their overlap. That
  comparison (correlation of weekly changes, level ratio) is the evidence behind
  each `fidelity` value.

### 4.2 Data providers and ingestion
- Provider interface: `get_history(source_symbol, start, end) -> DataFrame[obs_date, value]`.
  Implementations: `YahooProvider`, `FredProvider`, `CboeProvider`, `EwsCsvProvider`.
- Store values **raw, at native frequency**, exactly as the provider returns them,
  with `provider`, `obs_date`, `known_at` (earliest time the value was available,
  from the publication-lag rule), and `ingested_at`. Transforms (units, quote
  direction) and weekly alignment happen when features are built, not at ingestion,
  so a wrong transform never requires re-fetching.
- Labels live in their own table (`labels(label_set, week, value, source)`); the
  EWS `Y` column is label set `ews_v1`.
- Daily job: for each active instrument, fetch since the last stored date minus a
  5-day overlap (catches late revisions), validate, upsert into `prices`.
- Use adjusted close for equities/ETFs. Futures are front-month continuous; record that.
- Store dates in UTC as trading dates; keep the source's native calendar.
- Retries with backoff; one failed instrument must not fail the whole run.
  Log a per-run summary (fetched, skipped, failed, stale) to an `ingestion_runs` table.
- Staleness: if an instrument hasn't updated in N trading days, mark it stale and
  surface that in the API response rather than silently scoring on old data.
- After a successful run, publish a "data updated" event (Redis pub/sub is enough
  before Phase 8) so the scoring service can invalidate cached results (4.9.3).

### 4.3 Regime model
- **Frequency:** ingest daily, model weekly. Resample to `W-TUE` to stay aligned with
  the existing labels (the CSV's dates are Tuesdays). Make the anchor a config value.
- **Features:** only instruments the serving path can fetch. Use stationary
  transforms: log returns over 1/4/12 weeks, z-score vs. trailing 52-week window,
  realized volatility, yield changes in bp, spreads (e.g. 10y–3m). No raw levels,
  no era flags like `USD_Over_100`.
- **One feature function** (`features.build(prices_df) -> features_df`) shared by
  training and serving. Before Phase 8 this is the train/serve-skew guard; in Phase 8
  the same spec drives the streaming job and Feast.
- **Labels:** two targets, kept as separate models or heads:
  - `detect`: anomaly at week t (current label).
  - `predict`: any anomaly in weeks t+1..t+h (h configurable, default 4).
- **Evaluation:** expanding-window walk-forward across the full history, at minimum
  one fold per year from 2008 on. Report per fold and overall:
  - row-level precision/recall/F1 next to the always-positive and always-negative baselines
  - event-level: share of episodes flagged, lead time before onset (predict head),
    false alarms per year in calm periods
  - calibration: reliability curve, Brier score, and log loss
- **Explanations:** compute SHAP (or permutation contributions) against the final
  calibrated output, mapped back to canonical feature names a user can read.
  Explanations run in the scoring service, not Triton (4.9.2).
- **Artifacts:** `artifacts/<model_version>/` containing:
  - the sklearn pipeline (used by the scoring service for SHAP and as the parity reference)
  - `model.onnx` (served by Triton) and the parity-test result
  - `metadata.json`: feature list and order, training date range, data snapshot hash,
    library versions, metrics, decision threshold
  - a model card
  Register each in `model_versions`. Do not commit artifacts to git. Only load
  artifacts this app produced.
- Scores written to `regime_scores(week, model_version, head, probability, top_features)`.

### 4.4 Episode analyzer
- Build `anomaly_episodes` from contiguous runs of `Y=1` (merge runs separated by
  ≤ 2 calm weeks; configurable). Columns: id, start, end, peak week, optional name,
  optional type tag (credit, liquidity, inflation, geopolitical), label source.
- Per instrument per episode: return over the episode, max drawdown, peak-to-recovery
  time, correlation to the equity benchmark during the episode vs. the prior 52 calm weeks.
- Report per episode, never a single blended score. Show the count ("rose in 4 of 6").
- Insufficient history is an explicit result state, not a missing row.
- Results are cached in Redis per instrument (4.9.3).

### 4.5 News service
- Licensed news API only (provider is an open decision). Store headline, source,
  URL, published time, matched instruments/episode. No article bodies.
- Deduplicate syndicated copies (normalized title + time window).
- Cache per query; respect provider rate limits.
- Article text is untrusted input. It never goes into the narrator's prompt as
  instructions; if used at all, pass only headlines as quoted data.

### 4.6 LLM narrator
- Input: a JSON object of computed results (regime score, top features, episode stats).
- Output: plain-language summary. The prompt forbids introducing numbers or claims
  not in the input and forbids advice language.
- Post-check: every number in the output must appear in the input; on failure, fall
  back to a templated summary. Templated text is always available without the LLM.
- Narration doubles as the accessible text alternative for charts (4.8).

### 4.7 API gateway (FastAPI)
Public REST API, versioned under `/v1`. It owns auth, validation, and response shape;
it calls the scoring service over gRPC for anything model-related.
- `GET /regime/current` → latest score per head, model version, data freshness, top features
- `GET /regime/history?from&to` → weekly scores
- `GET /episodes` and `GET /episodes/{id}` → episode list and per-instrument behavior
- `GET /instruments`, `GET /groups` → tracked set and grouping
- `GET /instruments/{id}/episodes` → one instrument's behavior across all episodes
- `POST /uploads` → validate and score an uploaded CSV (see 5.2)
- `GET /news?episode_id|instrument_id`
- `GET /health` (liveness) and `GET /ready` (readiness: db, scoring service, last
  ingestion run, active model version)

Responses carry `model_version`, `data_as_of`, `cached` (bool), and per-instrument
`fidelity` so the UI can show proxies and staleness. Errors return structured JSON
with a stable code. gRPC errors map to HTTP status codes in one place.

### 4.8 Frontend requirements (stack TBD)
- Never encode meaning by color alone; pair color with text or icons.
- Every chart has a data-table toggle and a one-sentence text summary.
- Regime changes announced via a polite `aria-live` region, not every refresh.
- Confidence shown as words plus number ("Elevated stress, 72%"), with a short
  explanation of what the number means and a link to the model card.
- Full keyboard navigation, visible focus, `prefers-reduced-motion` respected,
  locale-aware number and date formatting.
- Persistent disclaimer: informational, not investment advice.

### 4.9 Serving layer (project requirement)
Requirement: serve the models through Triton behind a gRPC scoring service, with a
Redis cache for frequently queried results, deployed on Kubernetes.

#### 4.9.1 gRPC contract
Define `proto/scoring/v1/scoring.proto`; generate Python stubs in the build, don't
commit them. Initial RPCs:
- `ScoreCurrent(ScoreCurrentRequest{head}) -> ScoreResponse`
- `ScoreWindow(ScoreWindowRequest{head, rows}) -> ScoreResponse` (uploads, backtests)
- `ExplainScore(ExplainRequest{score_id}) -> Explanation`
- `GetInstrumentEpisodes(InstrumentRequest{instrument_id}) -> InstrumentEpisodeStats`

`ScoreResponse` carries probability, label at the model's threshold, model version,
`data_as_of`, top features, and `cached`. Rules: version the package (`scoring.v1`),
only add fields (never renumber or reuse tags), set deadlines on every client call,
and use the standard gRPC health-checking service for Kubernetes probes.

#### 4.9.2 Scoring service and Triton
Request flow: gateway → scoring service → Redis lookup → on miss: load prices,
`features.build()`, call Triton for the probability, compute SHAP locally, write
result to Redis and `regime_scores`, return.

- **Triton backend:** ONNX Runtime. Model repository layout:
  `model_repository/regime_detect/<version>/model.onnx` and
  `model_repository/regime_predict/<version>/model.onnx`, each with a
  `config.pbtxt` declaring input/output names, shapes, and dtypes.
- **Two-stage export:** first try exporting IsolationForest + stack + calibration as
  one ONNX graph. If conversion fails (likely spot: the `FrozenEstimator` calibration
  wrapper), export the stages as separate Triton models and chain them in the
  scoring service. Record which approach was used in `metadata.json`.
- **Parity:** the scoring service refuses to start if the Triton model version and
  the local sklearn artifact (used for SHAP) have different `model_version` values.
- **Model control:** Triton in explicit model-control mode; the scoring service
  loads the version named in `model_versions` as active. Rollback = change the
  active version.
- **Failure handling:** if Triton is unavailable, return a clear "scoring
  unavailable" error. Never fall back silently to a different model version.
  Retries with backoff on transient errors only.

#### 4.9.3 Redis cache
The model is market-level, so "frequently queried tickers" is implemented as
frequently queried **instruments** and results. Keys include the model version and
data timestamp so a stale entry can't be served after either changes.

| Key | Value | Invalidated by |
|---|---|---|
| `score:{model_version}:{head}:{data_as_of}` | latest regime score + explanation | new data, new model version |
| `episodes:instrument:{instrument_id}:{data_as_of}` | per-episode stats for one instrument | new data, episode rebuild |
| `episodes:list:{episode_build_id}` | episode list | episode rebuild |
| `upload:{upload_id}` | upload scoring result | TTL 24h |

Rules: always set a TTL as a backstop even when keys are versioned; use a short lock
to prevent many concurrent misses recomputing the same score; the app must work
correctly (just slower) if Redis is down. Track hit rate per key prefix.

#### 4.9.4 Kubernetes
Manifests under `deploy/k8s/` (Kustomize base + `local` overlay; Helm optional later).
Target: `kind` or `minikube`.

| Workload | Kind | Notes |
|---|---|---|
| api (FastAPI) | Deployment + Service + Ingress | stateless; readiness hits `/ready` |
| scorer (gRPC) | Deployment + Service | gRPC health probes; HPA on CPU |
| triton | Deployment + Service | model repo from a PersistentVolume or object storage |
| redis | Deployment or StatefulSet + Service | cache only, persistence not required |
| postgres | StatefulSet + PVC | or an external managed DB when deployed |
| ingestion | CronJob | daily schedule; concurrencyPolicy: Forbid |
| migrations | Job | runs Alembic before api/scorer roll out |

Config in ConfigMaps, credentials in Secrets (never in images or git), resource
requests and limits on every container, and rolling updates for api and scorer.
A model release = upload artifacts to the model repo, update the active version,
roll the scorer.

---

## 5. Data contracts and edge cases

### 5.1 Price data
Handle explicitly, with a test for each: missing days and holidays, provider gaps,
duplicate dates, splits/dividends (adjusted close), futures rolls and negative prices
(WTI 2020-04), quote-direction flips, discontinued series, symbol renames,
instruments with history shorter than the feature window.

### 5.2 CSV uploads
- Required: a date column (accept `Data`/`Date`/`date`) and columns that map to
  canonical IDs, either by canonical ID or by any known vendor symbol.
- Detect frequency; resample daily to weekly or reject with a clear message.
- Reject ambiguous date formats unless the user specifies one (`1/11/2000`).
- Enforce size and row limits; strip formula-leading cells (`=`, `+`, `-`, `@`) on display.
- After cleaning, require enough rows to fill the longest feature window.
- Uploads are scored with the active model and stored per upload; they are never
  used for training the shared model.
- Each failure returns a specific, human-readable reason.

### 5.3 Serving
Test explicitly: Triton down, Triton serving a different version than the scorer
expects, Redis down, Redis returning an entry for an old model version, scorer
deadline exceeded, concurrent cache misses for the same key, a model rollout while
requests are in flight.

---

## 6. Build phases

Each phase ends runnable with passing tests. Do not start a phase before the previous
one's acceptance criteria are met, unless the user says otherwise.

**Phase 0 — Repo hygiene** (`docs/phases/phase0-plan.md`) — **complete, pending PR**
- Merged the existing branches; untracked generated files and `artifacts/`.
- `docs/` + `CLAUDE.md`; `src/` layout; `pyproject.toml` + pinned `uv.lock`;
  top-level `tests/`; ruff; CI; README.
- Training and backtest results reproduce `main` exactly.

**Phase 1 — Symbol map and data layer** (`docs/phases/phase1-plan.md`)
- `config/instruments.yaml` covering all 42 EWS series and the 8 stress series,
  with fidelity flags, fallback priority, publication lag, and revision flags.
- Postgres schema via Alembic (instruments, symbol_map, prices, labels, ingestion runs).
- Providers for Yahoo, FRED, Cboe, and the EWS CSV; ingestion CLI (backfill, update).
- Symbol coverage report, including proxy-vs-Bloomberg comparison.
- `scripts/fetch_stress_data.py` retired once providers reproduce `stress_extra.csv`.
- No model changes: training keeps reading the CSV until Phase 2.
- Done when: backfill populates all available instruments, the EWS data in the
  database matches the CSV exactly, and the coverage report exists.

**Phase 2 — Model rebuild**
- Builds on `backtest.py`: same fold scheme, extended with event-level metrics
  (episodes caught, lead time, false alarms per calm year), the predict head, and
  calibration diagnostics. Target the known failures (2008 missed, 2020 false alarms).
- Shared stationary feature function reading from the database; candidate inputs
  include the Cboe/FRED stress series; calibrated output; SHAP on the final output;
  versioned artifacts.
- Retrain on the instruments actually available from Yahoo/FRED.
- ONNX export with a parity test, so Phase 5 isn't blocked by an unexportable model.
- Done when: evaluation report covers every fold, beats both trivial baselines at the
  event level, ONNX parity passes, and the report is saved alongside the artifact.
  Report honestly if the model doesn't beat the baselines.

**Phase 3 — Episodes**
- Build `anomaly_episodes`; per-instrument episode stats.
- Done when: every episode has stats or an explicit insufficient-history state.

**Phase 4 — API (in-process scoring)**
- FastAPI endpoints in 4.7 with an in-process `Scorer`; uploads per 5.2;
  Docker Compose for api + db + scheduler.
- Done when: endpoints have tests, including upload failure cases.

**Phase 5 — Serving architecture (required)**
Build in this order:
- 5a. gRPC contract and scoring service, still calling the model in-process; the
  gateway's `Scorer` becomes the gRPC client. Done when the API tests pass unchanged.
- 5b. Triton with the ONNX models; scoring service calls Triton; SHAP stays local.
  Done when Triton and local sklearn outputs match within tolerance on the test set.
- 5c. Redis cache per 4.9.3 with invalidation on new data and new model version.
  Done when the cache tests in 5.3 pass and hit rate is reported.
- 5d. Kubernetes manifests per 4.9.4 on `kind` or `minikube`. Done when the whole
  stack runs in the cluster, a model version rollout and rollback work without
  failed requests, and the ingestion CronJob runs.
- Report: per-stage latency (gateway, scorer, Triton, SHAP), p50/p95 end to end,
  cache hit rate, and cold vs. warm latency.

**Phase 6 — Narrator and news**
- Templated summaries first, then the LLM narrator with the number check; news service.

**Phase 7 — Frontend**
- Build against the requirements in 4.8.

**Phase 8 — Live system**
Build in this order; each step must work end to end before the next.

8a. **Real-time source and model.** Choose a real-time provider (section 9).
yfinance is unofficial, rate-limited, and not a streaming feed, so it can't be the
live source. Decide the live scoring frequency (e.g. hourly or 15-minute bars) and
train a model at that frequency; the weekly model can't score intraday ticks
meaningfully. Done when the live-frequency model passes the same walk-forward
evaluation standard as Phase 2 and exports to ONNX for Triton.

8b. **Kafka + replay.** Ingestion topic for ticks, a replay producer feeding stored
history at a compressed clock, and the live provider as a second producer.
Done when replayed history and live ticks land on the same topic in the same schema.

8c. **Stream features + Feast.** Flink or PySpark Structured Streaming job computing
the shared feature spec over windows, writing to Feast (online store for serving,
offline store for training). Done when features computed in the stream match the
batch `features.build()` output on replayed history within tolerance (skew test).

8d. **Streaming scores.** The scoring service reads features from Feast, scores via
the Phase 5 Triton/Redis path, and publishes results to a Kafka output topic and a
FastAPI WebSocket endpoint (`/v1/regime/stream`). Done when end-to-end latency from
tick to client is measured and reported.

8e. **FAISS similar-episodes.** Embed each historical anomaly week; `/v1/similar`
returns the nearest past situations to the current state. Done when results are
shown with their episode context, not just vectors.

8f. **Kubernetes additions.** Add Kafka (Strimzi), the stream job, and Feast to the
Phase 5 manifests. Done when a rolling model update happens without dropping the stream.

Metrics to report for Phase 8: throughput (events/sec), per-stage and end-to-end
latency, Redis hit rate, consumer lag, train/serve feature skew, and model metrics
at the live frequency.

---

## 7. Working rules for Claude

- Before starting a phase, write a phase plan (`docs/phases/phase<N>-plan.md`) based on
  an audit of the current code: findings with evidence, target layout, ordered tasks
  (one commit each), acceptance checks, and decisions with defaults. Get the user's
  go-ahead on the plan before executing it.
- Audit every branch, not just `main` (`git branch -a`, `git log main..origin/<branch>`).
- Work on a feature branch per phase or step; open a PR. Never push directly to `main`
  unless the user asks.
- Running code must not leave `git status` dirty. If it does, fix the ignore rules
  rather than committing the generated files.
- Don't commit generated files (`__pycache__`, artifacts, ONNX models, generated
  gRPC stubs, data snapshots, `.env`).
- Never commit API keys or credentials; read them from environment variables or
  Kubernetes Secrets.
- Write tests alongside code; every edge case in section 5 gets a test.
- Report metrics exactly as measured, including when they're worse than before.
  Always show the trivial baselines next to model metrics, and measure with the
  versions in `uv.lock`.
- If a task needs a decision listed in section 9, ask the user before choosing.
- If a request conflicts with section 1 scope, point that out before building it.
- Keep this file current: update section 3 after each phase and log decisions in section 8.

---

## 8. Decision log

| Date | Decision |
|---|---|
| 2026-10-13 | Scope narrowed to anomaly detection; no recommendations, suggestions, or valuation. |
| 2026-10-13 | Primary data source is Yahoo Finance; FRED as secondary; Bloomberg codes kept only as a mapping column. |
| 2026-10-13 | Instruments are referenced by canonical ID; vendor symbols live in the symbol map. |
| 2026-10-13 | Streaming/serving infra is kept, not dropped. Earlier phases follow the live-ready rules in 2.3. |
| 2026-10-13 | Project requirement: Triton behind a gRPC scoring service, Redis cache, Kubernetes deployment. These don't depend on live data, so they move to Phase 5, right after the API. Kafka, Flink/PySpark, Feast, and FAISS remain in the live phase (Phase 8). |
| 2026-10-13 | Redis caches frequently queried instruments and results (regime score, per-instrument episode stats), not per-ticker scores, because the model is market-level. |
| 2026-10-13 | SHAP explanations run in the scoring service using the sklearn artifact; Triton serves the ONNX model for predictions. |
| 2026-10-13 | Each phase gets its own plan file, approved by the user before execution. |
| 2026-10-13 | Branches `phase1-fixes` and `walk-forward-backtest` merged into `main`. The walk-forward backtest is the evaluation baseline; headline metrics come from it, not the single-year split. |
| 2026-10-13 | Cboe added as a data provider (implied-vol and skew indices), with Yahoo as fallback. |
| 2026-10-13 | Prices are stored raw at native frequency with point-in-time `known_at`; transforms and weekly alignment happen at feature time. Labels get their own table. |
| 2026-10-13 | Phase 0 completed on `phase-0/repo-hygiene-v2` (supersedes `phase-0/repo-hygiene`). `uv.lock` pinned to the `requirements.txt` versions; `requires-python` capped at `<3.14` because pinned shap 0.50.0 has no 3.14 build. Training and backtest outputs identical to `main`. |
| 2026-10-13 | Phase 1 plan written (`docs/phases/phase1-plan.md`). |

---

## 9. Open decisions (ask the user)

1. **Labels after 2021-04.** The CSV's `Y` stops in April 2021. Options: keep the model
   frozen on historical labels; define a rule-based label (e.g. drawdown + VIX
   thresholds) to extend it; or hand-label new episodes. This blocks retraining on recent data.
2. **Source and license of the CSV.** The repo is public. Needed to know whether the
   CSV can stay and whether the original labeling rule can be recovered.
3. **Replacement for `GTITL2YR`** and other unavailable series: FRED alternative
   (often monthly), a different stress proxy, or drop.
4. **Detect, predict, or both** shipped to users first.
5. **News API provider** and its licensing terms.
6. **Frontend stack.**
7. **Hosting target** (local cluster only, or a cloud Kubernetes service) and whether
   this is a portfolio project or a public product, which affects disclaimers and auth.
8. **Real-time data provider for Phase 8** (e.g. Alpaca, Polygon/Massive, or a
   broker API) and its cost and coverage of the instruments in the symbol map.
   Many macro series (yields, FX, futures) need a paid feed for real-time data.
9. **Live scoring frequency** (15-minute, hourly, daily) and how labels are defined
   at that frequency.
10. **Flink or PySpark Structured Streaming** for Phase 8c.
11. **Revised series (NFCI, possibly STLFSI).** Today's history includes later
    revisions, which leaks future information into backtests. Options: fetch
    first-release vintages from ALFRED (needs a free FRED API key), or accept the
    bias and flag it. Needed before Phase 2 uses these series as features.
12. **Cboe data terms.** Confirm that using Cboe's public index CSVs fits the
    project's purpose (portfolio vs. public product).
13. **Labels for 2020.** The 2020 anomaly weeks are interleaved with calm weeks
    (Feb–Nov); decide whether that's intended before tuning the model against the
    2020 false alarms.
