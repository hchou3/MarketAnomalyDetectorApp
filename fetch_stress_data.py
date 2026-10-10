"""
Download extra market-stress series and align them to the project's weekly
(Tuesday) dates, without look-ahead.

Sources (all free):
  - Cboe's official daily CSVs (preferred), with Yahoo Finance via yfinance as fallback
  - FRED weekly financial-stress indices (no API key needed)

Each series is fetched independently: if one fails, the error is printed and
the rest are still written.

Usage:
    pip install pandas yfinance certifi
    python fetch_stress_data.py "data/FinancialMarketData.xlsx - EWS.csv" data/stress_extra.csv
"""
import io
import ssl
import sys
import traceback
import urllib.request

import pandas as pd

CBOE_URLS = [
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv",
    "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/{}_History.csv",
]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# Cboe symbol -> Yahoo ticker (fallback)
CBOE_SERIES = {
    "VIX3M": "^VIX3M",   # 3-month implied vol
    "VIX9D": "^VIX9D",   # 9-day implied vol
    "VVIX": "^VVIX",     # volatility of VIX
    "SKEW": "^SKEW",     # tail-risk pricing
    "OVX": "^OVX",       # crude-oil implied vol
    "GVZ": "^GVZ",       # gold implied vol
}
# Weekly indices dated Friday, published the following Wed/Thu.
FRED_SERIES = {
    "NFCI": "NFCI",          # Chicago Fed National Financial Conditions Index (from 1971)
    "STLFSI": "STLFSI4",     # St. Louis Fed Financial Stress Index (from Dec 1993)
}
FRED_PUBLICATION_LAG_DAYS = 11  # a Friday-dated value is known by the Tuesday 11 days later


def _ssl_context():
    # Windows Python often lacks the CA bundle urllib needs; certifi ships one.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CTX = _ssl_context()


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"})
    with urllib.request.urlopen(req, timeout=60, context=SSL_CTX) as r:
        return r.read().decode("utf-8")


def _short(e):
    return f"{type(e).__name__}: {str(e)[:120]}"


def cboe_daily(symbol, yahoo):
    errors = []
    for template in CBOE_URLS:
        url = template.format(symbol)
        try:
            df = pd.read_csv(io.StringIO(_get(url)))
            df.columns = [c.strip().upper() for c in df.columns]
            close_col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
            s = pd.Series(pd.to_numeric(df[close_col], errors="coerce").values,
                          index=pd.to_datetime(df[df.columns[0]]), name=symbol)
            return s.dropna().sort_index(), "cboe"
        except Exception as e:
            errors.append(_short(e))
    import yfinance as yf
    h = yf.Ticker(yahoo).history(period="max", interval="1d", auto_adjust=False)
    if h.empty:
        raise RuntimeError(f"Cboe failed ({'; '.join(errors)}) and Yahoo returned no data")
    s = h["Close"].rename(symbol)
    s.index = s.index.tz_localize(None).normalize()
    return s.dropna().sort_index(), f"yahoo (cboe: {errors[0]})"


def fred_weekly(series_id, name):
    text = _get(FRED_URL.format(series_id))
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise RuntimeError(f"unexpected FRED response: {text[:200]!r}")
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0]), name=name)
    return s.dropna().sort_index(), "fred"


def main(ews_path, out_path):
    weeks = pd.read_csv(ews_path, usecols=["Data"], parse_dates=["Data"])["Data"].sort_values()
    out = pd.DataFrame(index=pd.DatetimeIndex(weeks, name="Data"))
    failed = []

    # Daily series: last close on or before each Tuesday (same convention as the EWS data)
    for sym, yahoo in CBOE_SERIES.items():
        try:
            s, src = cboe_daily(sym, yahoo)
        except Exception as e:
            print(f"{sym:7s} FAILED  {_short(e)}")
            failed.append(sym)
            continue
        print(f"{sym:7s} {src[:60]:60s} {s.index.min().date()} .. {s.index.max().date()}  {len(s)} days")
        aligned = s.reindex(s.index.union(out.index)).ffill().reindex(out.index)
        aligned[out.index < s.index.min()] = float("nan")  # never back-fill before the first value
        out[sym] = aligned

    # Weekly FRED series: only use values already published by that Tuesday
    for name, sid in FRED_SERIES.items():
        try:
            s, src = fred_weekly(sid, name)
        except Exception as e:
            print(f"{name:7s} FAILED  {_short(e)}")
            traceback.print_exc(limit=1)
            failed.append(name)
            continue
        print(f"{name:7s} {src:60s} {s.index.min().date()} .. {s.index.max().date()}  {len(s)} weeks")
        known = s.copy()
        known.index = known.index + pd.Timedelta(days=FRED_PUBLICATION_LAG_DAYS)
        merged = pd.merge_asof(
            out.index.to_frame(index=False),
            known.rename(name).rename_axis("known").reset_index(),
            left_on="Data", right_on="known", direction="backward",
        )
        out[name] = merged[name].values

    out.to_csv(out_path)
    print(f"\nWrote {out_path}: {len(out)} weeks x {out.shape[1]} series")
    if out.shape[1]:
        print("First week with data:")
        print(out.apply(lambda c: c.first_valid_index().date() if c.notna().any() else None).to_string())
    if failed:
        print(f"\nFailed: {', '.join(failed)} (see errors above)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
