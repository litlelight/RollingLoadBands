# E30 NYISO duration-weighted aggregation sensitivity

This retrospective sensitivity was defined after inspection of the original 2025 results. It does not replace the frozen equal-observation target or establish the operator's physical hourly energy. Each system-load observation is held forward until the next source timestamp, with intervals longer than 30 minutes left unbridged. UTC hours are integrated over elapsed seconds, then converted to local date/hour; the original 23/25-hour DST normalization is reused. At the dataset and long-gap boundaries, at most five minutes are filled with the adjacent observation.

## Input and invariance checks

- Original profiles regenerated from the current 96 raw ZIPs differ from the frozen E0 profiles by at most 0 MW.
- Alternative daily availability: 2915 days; DST-adjusted days: 16; retained feature dates: 2803.
- Regular 12-observation on-grid UTC hours: 63085; maximum alternative-minus-equal mean difference: 1.091393642e-11 MW.
- Long inter-observation gaps above 30 minutes: 1; see `long_timestamp_gaps.csv`.

## 2025 load and event sensitivity

- UTC hours compared: 8760; hours with absolute difference above 1e-6 MW: 945.
- Days with at least one changed hourly position: 316 of 365.
- Absolute hourly difference median / p95 / maximum: 0.0000 / 16.5727 / 136.0523 MW.
- Absolute daily-peak difference median / p95 / maximum: 0.0000 / 9.3074 / 46.7221 MW.
- Historical pre-2024 Q95: equal-observation 27091.3863 MW; duration-weighted 27091.3863 MW.
- Original 2024 validation events: 15; changed validation event dates at fixed threshold: 0.
- Original 2025 events: 17; duration-weighted events at original threshold: 17; at reestimated threshold: 17.
- Original peak-event days with an altered daily peak: 8 of 17; maximum absolute peak change on those days: 36.1846 MW.
- Event dates changed relative to original at fixed threshold: none.
- Event dates changed relative to original at reestimated threshold: none.

## Forecast and decision results

All alternative features, monthly Ridge fits, residual bands, and 2024 decision tuning were regenerated from the alternative target. No 2025 target was used in training or parameter selection for its own forecast. `forecast_band_comparison.csv` and `primary_decision_comparison.csv` contain the full numerical comparison. The original threshold is evaluated separately from a pre-2024 reestimated threshold so target sensitivity and threshold sensitivity are not conflated.

```text
                   scenario              method  ridge_mae_mw  daily_simultaneous_coverage_pct  daily_upper_coverage_pct  mean_width_mw
published_equal_observation standardized_max_90    610.656592                        84.383562                 92.328767    4366.972004
published_equal_observation    one_sided_max_90    610.656592                              NaN                 89.589041            NaN
          duration_weighted standardized_max_90    610.734902                        85.205479                 92.602740    4369.222591
          duration_weighted    one_sided_max_90    610.734902                              NaN                 89.041096            NaN
```

```text
                               scenario              method  fixed_threshold_mw setting_type  setting  TP  FP  FN  alerts  cost
            published_equal_observation         tuned_point         27091.38631     delta_mw   1900.0  17  11   0      28  11.0
            published_equal_observation two_sided_conformal         27091.38631       lambda      0.6  17  22   0      39  22.0
            published_equal_observation one_sided_conformal         27091.38631       lambda      0.7  17  20   0      37  20.0
   duration_weighted_original_threshold         tuned_point         27091.38631     delta_mw   1900.0  17  10   0      27  10.0
   duration_weighted_original_threshold two_sided_conformal         27091.38631       lambda      0.6  17  22   0      39  22.0
   duration_weighted_original_threshold one_sided_conformal         27091.38631       lambda      0.7  17  20   0      37  20.0
duration_weighted_reestimated_threshold         tuned_point         27091.38631     delta_mw   1900.0  17  10   0      27  10.0
duration_weighted_reestimated_threshold two_sided_conformal         27091.38631       lambda      0.6  17  22   0      39  22.0
duration_weighted_reestimated_threshold one_sided_conformal         27091.38631       lambda      0.7  17  20   0      37  20.0
```

The forward-hold calculation is one explicit sensitivity definition. Source timestamps do not independently certify whether an observation represents the start, end, or midpoint of a physical interval. Thus a small difference supports robustness to this specified alternative, not proof of duration-weighted physical truth. A large difference requires updating the manuscript's NYISO interpretation rather than selecting a favorable target.
