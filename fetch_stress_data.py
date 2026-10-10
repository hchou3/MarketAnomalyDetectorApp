import io
import sys
import urllib.request

import pandas as pd

CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices{}_History.csv"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

CBOE_SERIES = {
    "VIX3M": "^VIX3M",   # 3-month implied vol; history from Dec 2007
    "VIX9D": "^VIX9D",   # 9-day implied vol; history from 2011
    "VVIX": "^VVIX",     # volatility of VIX; history from Jan 2007
    "SKEW": "^SKEW",     # tail-risk pricing; history from 1990
    "OVX": "^OVX",       # crude-oil implied vol; history from 2007
    "GVZ": "^GVZ",       # gold implied vol; history from 2008
}

FRED_SERIES = {
    "NFCI": "NFCI",          # Chicago Fed National Financial Conditions Index (from 1971)
    "STLFSI": "STLFSI4",     # St. Louis Fed Financial Stress Index (from Dec 1993)
}
FRED_PUBLICATION_LAG_DAYS = 11  # a Friday-dated value is known by the Tuesday 11 days later

def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        return response.read().decode("utf-8")

def cboe_daily(symbol, yahoo):
    try:
        df = pd.read_csv(io.StringIO(_get(CBOE.format(symbol))), parse_dates=['Date'])
        df.columns = [c.strip().upper() for c in df.columns]
        date_col = df.columns[0]
        close_col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
        s = pd.Series(pd.to_numeric(df[close_col], errors="coerce").values, index=pd.to_datetime(df[date_col]), name=symbol)

        src = "cboe"
    except Exception as e:
        import yfinance as yf
        h = yf.Ticker(yahoo).history(period="max", interval="1d", auto_adjust=False)
        s = h["Close"].rename(symbol)
        s.index = s.index.tz_localize(None).normalize()
        src = f"yahoo (cboe failed: {type(e).__name__})"

    s = s.dropna().sort_index()
    print(f"{symbol:7s} {src:40s} {s.index.min().date()} .. {s.index.max().date()}  {len(s)} days")
    return s

def fred_weekly(series_id, name):
    df = pd.read_csv(io.StringIO(_get(FRED.format(series_id))))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0]), name=name).dropna()
    print(f"{name:7s} {'fred':40s} {s.index.min().date()} .. {s.index.max().date()}  {len(s)} weeks")
    return s

def main(ews_path, out_path):
    weeks = pd.read_csv(ews_path, usecols=["Data"], parse_dates=["Data"])["Data"].sort_values()
    out = pd.DataFrame(index=pd.DatetimeIndex(weeks, name="Data"))
 
    # Daily series: last close on or before each Tuesday (same convention as the EWS data)
    for sym, yahoo in CBOE_SERIES.items():
        s = cboe_daily(sym, yahoo)
        out[sym] = s.reindex(s.index.union(out.index)).ffill().reindex(out.index)
        out.loc[out.index < s.index.min(), sym] = float("nan")  # no back-filling before first value
 
    # Weekly FRED series: only use values already published by that Tuesday
    for name, sid in FRED_SERIES.items():
        s = fred_weekly(sid, name)
        s.index = s.index + pd.Timedelta(days=FRED_PUBLICATION_LAG_DAYS)  # date it became known
        out[name] = pd.merge_asof(out.reset_index(), s.rename(name).rename_axis("known").reset_index(),
                                  left_on="Data", right_on="known", direction="backward")[name].values
 
    out.to_csv(out_path)
    print(f"\nWrote {out_path}: {len(out)} weeks x {out.shape[1]} series")
    print("Coverage (first week with data):")
    print(out.apply(lambda c: c.first_valid_index().date() if c.notna().any() else None).to_string())

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])