# Data

## raw/ews_weekly_2000_2021.csv

**Original filename:** `FinancialMarketData.xlsx - EWS.csv`. The name format is how
Google Sheets exports one tab of a workbook, so this is the "EWS" tab of a workbook
called `FinancialMarketData.xlsx`. "EWS" most likely means Early Warning System.

**Rows:** 1,111 · **Dates:** 2000-01-11 to 2021-04-20 · **Frequency:** weekly, Tuesdays

**Columns:** `Data` (date), `Y` (label), and 42 series identified by Bloomberg codes:

| Group | Columns |
|---|---|
| Commodities | `XAU BGNL` (gold), `Cl1` (WTI front month), `CRY` (CRB commodity index), `BDIY` (Baltic Dry) |
| FX | `DXY` (dollar index), `JPY` (USD/JPY, higher = weaker yen), `GBP` (GBP/USD) |
| Volatility | `VIX` |
| US rates | `USGG30YR`, `GT10`, `USGG2YR`, `USGG3M`, `US0001M` (1M USD LIBOR) |
| European rates | `GTDEM30Y`, `GTDEM10Y`, `GTDEM2Y` (Germany), `GTITL30YR`, `GTITL10YR`, `GTITL2YR` (Italy), `EONIA` |
| Other rates | `GTJPY30YR`, `GTJPY10YR`, `GTJPY2YR` (Japan), `GTGBP30Y`, `GTGBP20Y`, `GTGBP2Y` (UK) |
| Bond indices (Bloomberg) | `LUMSTRUU`, `LMBITR`, `LUACTRUU`, `LF98TRUU`, `LG30TRUU`, `LP01TREU`, `EMUSTRUU`, `LF94TRUU` |
| Equity indices (MSCI) | `MXUS`, `MXEU`, `MXJP`, `MXBR`, `MXRU`, `MXIN`, `MXCN` |
| Macro | `ECSURPUS` (Citi US economic surprise index) |

The current model uses 5 of these (`DXY`, `JPY`, `VIX`, `GTITL2YR`, `XAU BGNL`) plus
features derived from them. The mapping of every column to live data sources is
Phase 1 work (`docs/phases/phase1-plan.md`).

**`Y`:** 1 = anomaly (stress regime), 0 = normal. 237 of 1,111 weeks are labelled 1.
The labeling rule was not documented with the dataset, and labels stop in April 2021
(architecture plan, open decisions 1 and 2).

**Source and license: unknown.** The Bloomberg codes and series such as `ECSURPUS`
indicate a Bloomberg terminal export. Bloomberg data generally can't be redistributed,
and **this repository is public**. Confirm the source and license, or remove the file
from the repo, before sharing the project more widely (open decision 2).

## raw/stress_extra.csv

**Frozen snapshot** fetched on or before 2026-10-13. Do not re-run the retired
`scripts/fetch_stress_data.py` (deleted in Phase 1 Step 7); use the database
providers instead (`mad-data backfill` then `mad-data parity-stress`).

Eight extra stress series aligned to the same 1,111 Tuesday dates:

| Column | Source | Notes |
|---|---|---|
| `VIX3M`, `VIX9D`, `VVIX`, `SKEW`, `OVX`, `GVZ` | Cboe daily index CSVs (Yahoo Finance fallback) | Last close on or before each Tuesday; blank before each series starts |
| `NFCI` | FRED (Chicago Fed National Financial Conditions Index) | Weekly, Friday-dated; a value is used only from the Tuesday 11 days later |
| `STLFSI` | FRED `STLFSI4` (St. Louis Fed Financial Stress Index) | Same publication-lag rule |

NFCI and STLFSI are marked `revised: true` in the instrument registry; their
published history is revised after initial release, so today's database values
may differ slightly from the snapshot. Use `mad-data parity-stress` to compare
the database against this file; differences in revised series are reported but
do not cause the command to fail.
