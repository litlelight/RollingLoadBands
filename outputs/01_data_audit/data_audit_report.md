# E0 Data Audit and Preprocessing Report

## Scope

This stage performed data ingestion, integrity checks, documented repair or imputation, DST normalization, exact-calendar feature construction, and information-boundary checks. No model was trained and no global predictor standardization was applied.

## ISO-NE

```text
Date range:        2003-01-01 - 2026-06-30
Raw days:          8582
Positions/day:     24
Hourly positions:  205968
2014 malformed:    removed = YES
2015 DST zero:     found = YES
Repaired value:    12378
Usable after 56d:  8526
Feature count:     233
Status:            PASS
```

The 2014 trailing fragment was removed only after confirming that the complete matching `0101151` record exists in the 2015 source file. The raw 2015-03-08 hour-ending 02 value was asserted to be 0 MW and the processed value was asserted to be 12,378 MW, the mean of its adjacent positions. Raw archives were not modified.

## NYISO

```text
Date range:        2018-01-01 - 2025-12-31
Calendar days:     2922
Official gap days: 7
Available days:    2915
Gap:               2021-04-01 - 2021-04-07
First valid target after gap: 2021-06-03
Expected full-feature targets: 2803
Zones:             11
Short missing timestamps: 15
Imputed zone cells: 165
Non-grid source timestamps: 16131
DST adjusted days: 16
DST timezone preserved: YES
Feature count:     233
Status:            PASS
```

The seven-day P-58B gap was not filled or bridged. Complete 56-day feature history is unavailable from 2021-04-01 through 2021-06-02, so the first valid post-gap target is 2021-06-03. Fifteen short timestamp gaps affecting all 11 zones were linearly interpolated zone by zone only when valid observations existed on both sides and the enclosing gap was at most 30 minutes. The source also contains 16131 reported timestamps that are not exactly aligned to a five-minute clock; they were retained and each reported observation received equal weight in its local-hour arithmetic mean. EDT and EST were converted to distinct UTC instants before aggregation. Spring-forward days were normalized by interpolating the missing local hour; fall-back days were normalized by averaging the two complete repeated-hour means.

## Feature boundary

The 233 predictors comprise 205 load-history features and 28 calendar features:

- 144 hourly load-profile values at lags 1, 2, 7, 14, 21, and 28 days.
- 21 lagged daily summaries for lags 1, 7, and 28 days.
- 28 rolling means of seven daily summaries over 7, 14, 28, and 56 days.
- 12 rolling standard deviations for daily mean, maximum, and range over the same four windows.
- 28 target-date calendar variables.

All load lookups use exact calendar dates. Every manifest entry is marked as known at the forecast origin. The saved feature tables contain raw, unstandardized predictors; scaling must be fitted separately inside each future model-training window.

## Gate decision

E0 overall status: **PASS**

E1 may begin because all mandatory E0 assertions passed.
