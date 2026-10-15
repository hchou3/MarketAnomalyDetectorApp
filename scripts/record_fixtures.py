"""Record fixture responses for offline provider tests.

Run this script once with live internet access to populate tests/fixtures/.
The recorded files are committed to the repo so CI can run provider tests
offline without hitting any external service.

Usage:
    uv run python scripts/record_fixtures.py               # all providers
    uv run python scripts/record_fixtures.py --provider fred
    uv run python scripts/record_fixtures.py --provider cboe
    uv run python scripts/record_fixtures.py --provider yahoo

ALFRED publication-lag measurement (optional):
    FRED_API_KEY=<key> uv run python scripts/record_fixtures.py --provider fred
    When FRED_API_KEY is set, the script also queries ALFRED for the real
    first-publication lag of DEXJPUS and DEXUSUK and prints a summary.
    This is the primary evidence for the publication_lag_days values in
    config/instruments.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from datetime import date, timedelta
from pathlib import Path

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from market_anomaly_detector.providers._http import http_get
from market_anomaly_detector.registry import load_registry

REPO_ROOT = Path(__file__).parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"

# ── Helpers ────────────────────────────────────────────────────────────────────


def _collect_symbols(provider: str) -> list[str]:
    """Return all non-EWS symbols for *provider* from the instrument registry."""
    instruments = load_registry()
    symbols = []
    for inst in instruments:
        for src in inst.sources:
            if src.provider == provider and src.symbol and src.fidelity.value != "unavailable":
                # bloomberg_ews is not a live provider; skip it here
                if provider != "bloomberg_ews":
                    symbols.append(src.symbol)
    return sorted(set(symbols))


def _short_err(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:120]}"


# ── ALFRED publication-lag measurement ────────────────────────────────────────


def _measure_alfred_lag(series_id: str, api_key: str) -> dict:
    """Query ALFRED for the first-publication lag of *series_id*.

    Returns a dict with keys: n, min, median, p95, max (all in calendar days).
    """
    since = (date.today() - timedelta(days=3 * 365)).isoformat()
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}"
        f"&api_key={api_key}"
        f"&realtime_start={since}"
        "&realtime_end=9999-12-31"
        f"&observation_start={since}"
        "&file_type=json"
    )
    text = http_get(url)
    data = json.loads(text)

    # For each obs_date, keep only the earliest realtime_start (first publication).
    first_pub: dict[date, date] = {}
    for obs in data.get("observations", []):
        if obs["value"] == ".":
            continue
        obs_date = date.fromisoformat(obs["date"])
        rt = date.fromisoformat(obs["realtime_start"])
        if obs_date not in first_pub or rt < first_pub[obs_date]:
            first_pub[obs_date] = rt

    if not first_pub:
        return {"n": 0}

    lags = sorted((rt - d).days for d, rt in first_pub.items())
    return {
        "n": len(lags),
        "min": min(lags),
        "median": statistics.median(lags),
        "p95": lags[int(0.95 * len(lags))],
        "max": max(lags),
    }


# ── Provider-specific recording functions ─────────────────────────────────────


def record_fred(api_key: str | None) -> None:
    out_dir = FIXTURES / "fred"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = _collect_symbols("fred")
    print(f"\n── FRED ({len(symbols)} symbols) ──────────────────────────────")

    for sym in symbols:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sym}"
        try:
            text = http_get(url)
            path = out_dir / f"{sym}.csv"
            path.write_text(text, encoding="utf-8")
            rows = text.count("\n") - 1  # subtract header
            print(f"  {sym:<20s} {rows:>6d} rows  → {path.relative_to(REPO_ROOT)}")
        except Exception as exc:
            print(f"  {sym:<20s} FAILED  {_short_err(exc)}")

    # ALFRED publication-lag measurement
    fx_series = ["DEXJPUS", "DEXUSUK"]
    if api_key:
        print("\n── ALFRED publication-lag measurement ──────────────────")
        print("  (based on the last 3 years; lag = first realtime_start − obs_date)\n")
        for sid in fx_series:
            try:
                stats = _measure_alfred_lag(sid, api_key)
                if stats["n"] == 0:
                    print(f"  {sid}: no data returned")
                else:
                    print(
                        f"  {sid}: n={stats['n']}, "
                        f"min={stats['min']}d, "
                        f"median={stats['median']}d, "
                        f"p95={stats['p95']}d, "
                        f"max={stats['max']}d"
                    )
            except Exception as exc:
                print(f"  {sid}: ALFRED query failed — {_short_err(exc)}")
        print("\n  instruments.yaml uses publication_lag_days: 10 as a conservative floor.")
    else:
        print(
            "\n  Set FRED_API_KEY to also query ALFRED for the real FX publication lag"
            f"\n  (series: {', '.join(fx_series)})."
        )


def record_cboe() -> None:
    out_dir = FIXTURES / "cboe"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = _collect_symbols("cboe")
    cdns = [
        "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv",
        "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv",
    ]
    print(f"\n── Cboe ({len(symbols)} symbols) ──────────────────────────────")

    for sym in symbols:
        text = None
        errors = []
        for template in cdns:
            url = template.format(sym=sym)
            try:
                text = http_get(url)
                break
            except Exception as exc:
                errors.append(_short_err(exc))
        if text is None:
            print(f"  {sym:<10s} FAILED  {'; '.join(errors)}")
            continue
        path = out_dir / f"{sym}_History.csv"
        path.write_text(text, encoding="utf-8")
        rows = text.count("\n") - 1
        print(f"  {sym:<10s} {rows:>6d} rows  → {path.relative_to(REPO_ROOT)}")


def record_yahoo() -> None:
    import yfinance as yf

    out_dir = FIXTURES / "yahoo"
    out_dir.mkdir(parents=True, exist_ok=True)

    instruments = load_registry()
    # Build symbol → auto_adjust mapping from the registry
    symbol_meta: dict[str, bool] = {}
    for inst in instruments:
        for src in inst.sources:
            if src.provider == "yahoo" and src.symbol:
                symbol_meta[src.symbol] = src.auto_adjust

    print(f"\n── Yahoo ({len(symbol_meta)} symbols) ──────────────────────────────")

    for sym, auto_adjust in sorted(symbol_meta.items()):
        try:
            df = yf.Ticker(sym).history(period="max", auto_adjust=auto_adjust)
            if df.empty:
                print(f"  {sym:<15s} EMPTY — skipped")
                continue
            # Strip timezone before saving so the CSV has plain datetime strings
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            fname = sym.replace("/", "_")
            path = out_dir / f"{fname}.csv"
            df.to_csv(path)
            print(
                f"  {sym:<15s} auto_adjust={str(auto_adjust):<5s} "
                f"{len(df):>6d} rows  → {path.relative_to(REPO_ROOT)}"
            )
        except Exception as exc:
            print(f"  {sym:<15s} FAILED  {_short_err(exc)}")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--provider",
        choices=["fred", "cboe", "yahoo"],
        help="Record only this provider (default: all).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("FRED_API_KEY")
    providers = [args.provider] if args.provider else ["fred", "cboe", "yahoo"]

    for p in providers:
        if p == "fred":
            record_fred(api_key)
        elif p == "cboe":
            record_cboe()
        elif p == "yahoo":
            record_yahoo()

    print("\nDone. Commit the new/updated files under tests/fixtures/.")


if __name__ == "__main__":
    main()
