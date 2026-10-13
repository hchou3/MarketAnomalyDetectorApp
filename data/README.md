# Data

## raw/ews_weekly_2000_2021.csv

**Original filename:** `FinancialMarketData.xlsx - EWS` (exported from a Google Sheets workbook)

**Rows:** 1,111  
**Date range:** 2000-01-11 to 2021-04-20  
**Frequency:** Weekly (Tuesday dates)

### Columns

| Column | Bloomberg code | Description |
|--------|---------------|-------------|
| Data | — | Date (YYYY-MM-DD) |
| XAU BGNL | XAU BGNL | Gold spot price (USD/troy oz, BGN) |
| VIX | VIX Index | CBOE Volatility Index |
| DXY | DXY Curncy | US Dollar Index |
| GTITL2YR | GTITL2YR Govt | Italy 2-year government bond yield (%) |
| EONIA | EONIA Index | Euro Overnight Index Average rate (%) |
| Y | — | Anomaly label: 1 = stress regime, 0 = normal |

### Notes

- Bloomberg codes are kept as column headers; live data ingestion maps them via the
  instrument symbol map (see `docs/architecture-plan.md` section 2.2).
- The `Y` column labels are binary (0/1). The labeling rule is unknown — it was not
  documented with the dataset. This is an open decision (see architecture-plan section 9,
  item 2).
- `GTITL2YR` has limited availability outside Bloomberg; a FRED substitute has not yet
  been selected (open decision 3).

### License and origin

**Unknown.** This file appears to have been exported from a Bloomberg terminal worksheet.
It must not be redistributed until the source and license are confirmed. It is currently
kept in the repo as a convenience while the project is non-public. Revisit before making
the repository public (architecture-plan open decision 2).
