# Market Anomaly Detector — Project Build Spec

## What this is

An upgrade of the existing `AnomalyDetection` project (Pandas, Scikit-Learn, TensorFlow, OpenAI, 4 models: decision tree, random forest, isolation forest, logistic regression, 84–87% F1) into a real-time, production-shaped system. Goal: take live or uploaded stock data, run it through an ensemble anomaly pipeline, and return a prediction with a confidence score, served through an actual API rather than a notebook.

This doc is meant to be dropped into a Claude Project (or Claude Code) as the working brief. Build in the phases below — each phase should produce something runnable before moving to the next.

---

## Core user flow

1. User uploads a CSV of historical price/volume data, or points the app at a ticker for live data.
2. System ingests, normalizes, and feature-engineers the data.
3. Data passes through a two-stage anomaly pipeline (unsupervised filter → supervised classification).
4. User gets back: anomaly flag, confidence score, and the SHAP-driven "why" behind the flag.
5. Optional: live-streaming mode where new ticks are scored continuously.

---

## Architecture overview

```
[Data Source]
  ├─ Live feed (websocket / polling API, e.g. Alpaca, Polygon, or yfinance for a free tier)
  └─ CSV upload
        │
        ▼
[Kafka] — ingestion topic, decouples source from processing
        │
        ▼
[Flink or PySpark Structured Streaming] — feature engineering, windowed aggregates
   (rolling volatility, volume z-scores, momentum features, etc.)
        │
        ▼
[Feast] — feature store; serves the same features consistently for training and
   real-time inference, avoids train/serve skew
        │
        ▼
[Stage 1: Isolation Forest] — fast unsupervised candidate filter
        │  (only flagged candidates proceed — cuts inference cost on the
        │   heavier supervised stage)
        ▼
[Stage 2: Stacked ensemble] — decision tree + random forest + logistic
   regression as base learners, meta-model on top (sklearn StackingClassifier,
   or a small PyTorch MLP if you want the ensembling itself to be learned)
        │
        ▼
[Redis] — cache recent feature vectors + recent predictions per ticker,
   so the live view doesn't recompute from scratch every tick
        │
        ▼
[Triton Inference Server] — serves the trained models behind a standard
   inference API; enables versioning and swapping models without redeploying
   the app
        │
        ▼
[gRPC] — internal service-to-service contract between the API layer and
   Triton, and between the ingestion service and the scoring service
        │
        ▼
[FastAPI backend] — public REST/WebSocket API: upload endpoint, live-score
   endpoint, historical-backtest endpoint
        │
        ▼
[Frontend] — upload UI, live ticker dashboard, confidence score + SHAP
   explanation panel
```

**FAISS** — use this for a "similar past anomalies" feature: embed each anomaly event (feature vector) and let users query "show me past anomalies that looked like this one." Nice differentiator, not core to the pipeline.

**Kubernetes** — containerize each service (ingestion, feature engineering, scoring, API) and deploy as separate pods. Gives you a real reason to talk about orchestration, scaling the scoring service independently from the ingestion service, and rolling updates when you retrain a model.

---

## Tech stack mapping (why each piece earns its place)

| Tool | Role | Why it's not just resume-padding |
|---|---|---|
| Kafka | Ingestion buffer | Decouples live data source from processing; lets you replay historical data at a compressed clock for backtesting |
| PySpark or Flink | Feature engineering | Windowed/rolling features (volatility, z-scores) need stream-aware computation, not just pandas `.rolling()` |
| Feast | Feature store | Guarantees the exact same feature computation at train time and serve time |
| Redis | Cache | Sub-second lookups for "what's this ticker's current state" without hitting the feature store every request |
| Triton | Model serving | Real model versioning/hot-swapping instead of a pickled model loaded in a Flask route |
| gRPC | Internal API contract | Typed, fast internal calls between services — standard in real ML infra |
| FAISS | Similarity search | Powers the "similar past anomalies" feature via vector search |
| Kubernetes | Deployment | Each service scales independently; matches how this would actually be run in production |
| PyTorch | Optional meta-model | If you want the ensemble combiner itself to be a learned model rather than sklearn's stacking classifier |

---

## Build phases

### Phase 1 — Get the ensemble working locally (no infra yet)
- [ ] Re-implement the 4-model pipeline as a `StackingClassifier` (or manual two-stage cascade: Isolation Forest filters, then the 3 supervised models vote/stack).
- [ ] Add a confidence score: for the meta-model, use `predict_proba` rather than a hard label. Calibrate it (`CalibratedClassifierCV`) so "70% confidence" actually means something.
- [ ] Automate SHAP so it reruns per prediction, not just once at training time — this is what powers the "why was this flagged" output.
- [ ] Wrap it in a single Python function: `score(df) -> {anomaly: bool, confidence: float, top_features: [...]}`.

### Phase 2 — Make it a real service
- [ ] FastAPI app with two endpoints: `/upload` (CSV in, batch scores out) and `/score/{ticker}` (live score).
- [ ] Redis cache for repeated queries on the same ticker within a short window.
- [ ] Containerize with Docker.

### Phase 3 — Streaming
- [ ] Kafka topic for incoming ticks (start with a replay script feeding historical data at a compressed clock — you don't need a live paid data feed to demonstrate this).
- [ ] PySpark Structured Streaming or Flink job consuming from Kafka, computing rolling features, writing to Feast.
- [ ] Scoring service reads from Feast + Redis, publishes results to a Kafka output topic or WebSocket.

### Phase 4 — Serving infra
- [ ] Export models to a Triton-compatible format (ONNX for sklearn models via `skl2onnx`, or TorchScript if you built a PyTorch meta-model).
- [ ] Stand up Triton, point the FastAPI scoring endpoint at it via gRPC instead of loading the model in-process.

### Phase 5 — FAISS similarity search
- [ ] Embed each historical anomaly's feature vector (can just be the raw scaled feature vector, or a small learned embedding).
- [ ] Build a FAISS index; add a `/similar/{event_id}` endpoint returning the nearest past anomalies.

### Phase 6 — Kubernetes
- [ ] Write manifests (or a Helm chart) for: ingestion service, feature/streaming job, scoring service, FastAPI gateway, Redis, Kafka (or use a managed/dev Kafka via `strimzi` operator).

- [ ] Get it running on a local cluster (`kind` or `minikube`) — doesn't need to be cloud-deployed to be legitimate for the resume, just needs to actually run.

---

## Metrics to track (for both the app and the resume bullet)

- F1 / precision / recall of the ensemble vs. the original best single model (Isolation Forest at 84–87% F1) — quantify the lift.
- Inference latency per score request (end-to-end, and broken down by stage).
- Cache hit rate on Redis for live queries.
- Throughput of the streaming pipeline (events/sec processed).
- Reduction in false positive rate if you add confidence thresholding (only surface predictions above X% confidence).

---

## Suggested resume bullet shape once built

Don't write this yet — write it after you've actually built and measured it. But structurally, aim for something like:

> "Built a real-time market anomaly detection system streaming live data through Kafka and Flink into a two-stage ensemble (Isolation Forest filter → stacked classifier), served via Triton/gRPC and deployed on Kubernetes; improved F1 from 84% (single model) to [X]% and cut inference latency to [Y]ms per request."

Fill in real numbers only once you've measured them — don't estimate.

---

## Suggested build order if time-constrained

If you don't get through all 6 phases, stop after Phase 2 or 3 — a working stacked ensemble served through a real API with Redis caching is already a legitimate, demoable upgrade. Phases 4–6 (Triton, FAISS, Kubernetes) are what take it from "solid project" to "looks like production ML infra," but they're additive, not required for the core story to be true.
