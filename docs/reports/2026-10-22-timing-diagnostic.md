# FX / Equity Timing Diagnostic

*Generated 2026-10-22. Shifts applied to live obs_dates before aligning to EWS Tuesdays.*

A **positive shift (+1 bd)** moves the live date forward (e.g., a Thursday close
treated as Friday), which may align better with an EWS snapshot taken at a later
point in the week.

## Correlation at obs_date Shifts

| Instrument | Provider | Symbol | EWS Reference | corr(−1 bd) | corr(0) | corr(+1 bd) | Best Shift | Best Corr | Note |
|---|---|---|---|---|---|---|---|---|---|
| gbp_usd | yahoo | GBPUSD=X | gbp_usd | 0.859 | 0.889 | 0.764 | +0 | 0.889 | no shift improves correlation |
| usd_jpy | yahoo | JPY=X | usd_jpy | 0.803 | 0.883 | 0.686 | +0 | 0.883 | no shift improves correlation |
| gbp_usd_fred | fred | DEXUSUK | gbp_usd | 0.809 | 0.989 | 0.785 | +0 | 0.989 | no shift improves correlation |
| msci_japan | yahoo | ^N225 | msci_japan | 0.765 | 0.971 | 0.765 | +0 | 0.971 | no shift improves correlation |
| msci_japan_n225 | yahoo | ^N225 | msci_japan | 0.765 | 0.971 | 0.765 | +0 | 0.971 | no shift improves correlation |

## Interpretation

- **corr ≥ 0.98** at a non-zero shift → candidate for obs_date/known_at correction.
- **corr ≥ 0.80** → proxy-grade; timing adjustment optional.
- **corr < 0.80** → below proxy threshold; consider replacing the source.

## Finding

No whole-day shift (±1 business day) improves any instrument's correlation with
the EWS reference.  Yahoo FX sources (gbp_usd, usd_jpy) show proxy-grade
correlation (~0.88–0.89) due to the WM/Reuters 4 pm London fixing vs Bloomberg
snapshot timing difference; this is not a calendar-offset issue and no obs_date
or known_at correction is applied.  FRED FX sources (gbp_usd_fred, usd_jpy_fred)
are gap-fill fallbacks only and are not corrected.
