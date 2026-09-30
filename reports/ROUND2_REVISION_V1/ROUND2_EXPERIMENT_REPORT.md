# ROUND2_REVISION_V1 — second-round reviewer experiments

These are retrospective analyses of frozen historical out-of-sample forecasts. The 90-day rows retain
the inherited reference setting. Selecting a window from 2024 is a newly specified retrospective
exercise, not an original prespecified protocol or an untouched 2025 confirmation.

## Gates and scope

Frozen ISO-NE hashes, both original 2025 coverage counts, and primary decision costs passed.
All 1050 decision-matrix rows were independently reconciled with 365 daily records each.
The new 90-day band paths agree with the prior paths to within 1e-6 MW.
Zero-step projected ACI agrees with the ordinary 90-day two-sided path.
Runtime: 22.6 s. Seed: 20260917.

## Calibration-window decision sensitivity

| dataset | method | window_days | selected_parameter | validation_cost | test_TP | test_FP | test_FN | test_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISO_NE | direct_peak_w90 | 90.000 | 0.800 | 39.000 | 16 | 10 | 1 | 20.000 |
| ISO_NE | direct_peak_w365 | 365.000 | 0.900 | 57.000 | 14 | 6 | 3 | 36.000 |
| ISO_NE | one_sided_w30 | 30.000 | 0.700 | 35.000 | 17 | 19 | 0 | 19.000 |
| ISO_NE | one_sided_w90 | 90.000 | 0.700 | 29.000 | 17 | 16 | 0 | 16.000 |
| ISO_NE | one_sided_w365 | 365.000 | 1.000 | 26.000 | 16 | 15 | 1 | 25.000 |
| ISO_NE | tuned_point | NA | 2350.000 | 26.000 | 16 | 14 | 1 | 24.000 |
| ISO_NE | two_sided_w30 | 30.000 | 0.700 | 37.000 | 17 | 32 | 0 | 32.000 |
| ISO_NE | two_sided_w90 | 90.000 | 0.700 | 30.000 | 17 | 17 | 0 | 17.000 |
| ISO_NE | two_sided_w365 | 365.000 | 1.000 | 29.000 | 17 | 19 | 0 | 19.000 |
| NYISO | direct_peak_w90 | 90.000 | 1.000 | 12.000 | 15 | 11 | 2 | 31.000 |
| NYISO | direct_peak_w365 | 365.000 | 1.000 | 36.000 | 14 | 8 | 3 | 38.000 |
| NYISO | one_sided_w30 | 30.000 | 0.500 | 22.000 | 16 | 17 | 1 | 27.000 |
| NYISO | one_sided_w90 | 90.000 | 0.700 | 17.000 | 17 | 20 | 0 | 20.000 |
| NYISO | one_sided_w365 | 365.000 | 0.900 | 15.000 | 16 | 11 | 1 | 21.000 |
| NYISO | tuned_point | NA | 1900.000 | 12.000 | 17 | 11 | 0 | 11.000 |
| NYISO | two_sided_w30 | 30.000 | 0.500 | 24.000 | 17 | 24 | 0 | 24.000 |
| NYISO | two_sided_w90 | 90.000 | 0.600 | 15.000 | 17 | 22 | 0 | 22.000 |
| NYISO | two_sided_w365 | 365.000 | 0.800 | 15.000 | 17 | 14 | 0 | 14.000 |

Full 30/60/90/180/270/365-day tables for both systems, both band families, and all 25
threshold/cost settings are in E25/window_decision_matrix.csv. The 2024 joint window/
lambda selection rows are in E25/retrospective_window_selection_2024.csv.

### Window-plus-lambda choices using 2024 cost

| dataset | method | window_days | selected_parameter | validation_cost | test_cost |
| --- | --- | --- | --- | --- | --- |
| ISO_NE | one_sided_w60 | 60.000 | 0.600 | 25.000 | 21.000 |
| ISO_NE | two_sided_w270 | 270.000 | 1.000 | 26.000 | 20.000 |
| NYISO | one_sided_w270 | 270.000 | 0.900 | 12.000 | 22.000 |
| NYISO | two_sided_w270 | 270.000 | 0.800 | 12.000 | 12.000 |

These selections were formulated for the revision after the 2025 record had been inspected.

## Sequential and direct-daily-maximum baselines

| dataset | method | period | days | joint_coverage | upper_coverage | peak_upper_coverage | mean_width_mw | interval_score_mw | mean_peak_margin_mw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISO_NE | aci_eta0 | 2025 | 365 | 0.849 | 0.890 | NA | 4943.730 | 5189.695 | NA |
| ISO_NE | one_sided_w90 | 2025 | 365 | NA | 0.852 | NA | NA | NA | NA |
| ISO_NE | two_sided_w90 | 2025 | 365 | 0.849 | 0.890 | NA | 4943.730 | 5189.695 | NA |
| ISO_NE | direct_peak_w365 | 2025 | 365 | NA | NA | 0.890 | NA | NA | 1180.578 |
| ISO_NE | direct_peak_w90 | 2025 | 365 | NA | NA | 0.890 | NA | NA | 1346.245 |
| NYISO | aci_eta0 | 2025 | 365 | 0.844 | 0.923 | NA | 4366.972 | 4719.910 | NA |
| NYISO | one_sided_w90 | 2025 | 365 | NA | 0.896 | NA | NA | NA | NA |
| NYISO | two_sided_w90 | 2025 | 365 | 0.844 | 0.923 | NA | 4366.972 | 4719.910 | NA |
| NYISO | direct_peak_w365 | 2025 | 365 | NA | NA | 0.896 | NA | NA | 1242.392 |
| NYISO | direct_peak_w90 | 2025 | 365 | NA | NA | 0.899 | NA | NA | 1255.108 |

ACI is a projected sequential alpha-update baseline with the same archived Ridge forecasts
and 90-day maximum score. Its step-size grid and 2024-only retrospective selection are saved.
The full ACI step-size comparison at historical Q95 and cost ratio 10 is:

| dataset | method | chosen_from_2024_coverage | joint_coverage | mean_width_mw | interval_score_mw | boundary_days | selected_parameter | validation_cost | test_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ISO_NE | aci_eta0 | False | 0.849 | 4943.730 | 5189.695 | 0.000 | 0.700 | 30.000 | 17.000 |
| ISO_NE | aci_eta0.002 | False | 0.890 | 5518.248 | 5641.322 | 0.000 | 0.600 | 29.000 | 21.000 |
| ISO_NE | aci_eta0.005 | False | 0.893 | 5533.444 | 5658.090 | 0.000 | 0.600 | 35.000 | 21.000 |
| ISO_NE | aci_eta0.01 | False | 0.896 | 5416.194 | 5545.289 | 0.000 | 0.600 | 36.000 | 19.000 |
| ISO_NE | aci_eta0.02 | True | 0.896 | 5205.361 | 5373.279 | 1.000 | 0.600 | 36.000 | 19.000 |
| ISO_NE | aci_eta0.05 | False | 0.890 | 4841.991 | 5045.548 | 9.000 | 0.500 | 33.000 | 17.000 |
| NYISO | aci_eta0 | False | 0.844 | 4366.972 | 4719.910 | 0.000 | 0.600 | 15.000 | 22.000 |
| NYISO | aci_eta0.002 | False | 0.885 | 5093.011 | 5278.951 | 0.000 | 0.500 | 14.000 | 22.000 |
| NYISO | aci_eta0.005 | False | 0.893 | 5413.709 | 5555.100 | 0.000 | 0.500 | 15.000 | 25.000 |
| NYISO | aci_eta0.01 | False | 0.899 | 5277.067 | 5392.295 | 0.000 | 0.500 | 15.000 | 25.000 |
| NYISO | aci_eta0.02 | True | 0.901 | 5520.457 | 5625.528 | 0.000 | 0.600 | 21.000 | 35.000 |
| NYISO | aci_eta0.05 | False | 0.893 | 5015.312 | 5173.100 | 7.000 | 0.600 | 20.000 | 28.000 |

ISO_NE: 2024-selected aci_eta0.02 has 2025 joint coverage 89.59%, width 5205 MW and cost 19; ordinary rolling 90-day has 84.93%, 4944 MW and cost 17.
NYISO: 2024-selected aci_eta0.02 has 2025 joint coverage 90.14%, width 5520 MW and cost 35; ordinary rolling 90-day has 84.38%, 4367 MW and cost 22.

Direct-peak calibration uses historical residuals of the daily predicted maximum and does not
produce a 24-position profile band.

## Full decision-matrix pattern relative to tuned point

| dataset | method | operating_settings | lower_cost | same_cost | higher_cost |
| --- | --- | --- | --- | --- | --- |
| ISO_NE | two_sided_w90 | 25 | 9 | 2 | 14 |
| ISO_NE | one_sided_w90 | 25 | 9 | 10 | 6 |
| NYISO | two_sided_w90 | 25 | 5 | 8 | 12 |
| NYISO | one_sided_w90 | 25 | 4 | 8 | 13 |

These 25 settings are correlated scenarios rather than 25 independent replications.

### Peak-event denominators by threshold

| dataset | threshold_spec | evaluation_events |
| --- | --- | --- |
| ISO_NE | historical_q90 | 27 |
| ISO_NE | historical_q95 | 17 |
| ISO_NE | historical_q97.5 | 9 |
| ISO_NE | historical_q99 | 4 |
| ISO_NE | rolling_365d_95 | 19 |
| NYISO | historical_q90 | 32 |
| NYISO | historical_q95 | 17 |
| NYISO | historical_q97.5 | 8 |
| NYISO | historical_q99 | 4 |
| NYISO | rolling_365d_95 | 22 |

## Calendar and resampling diagnostics

| dataset | month | method | days | events | joint_covered_days | upper_covered_days | paired_cost_difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ISO_NE | 3 | two_sided_w90 | 31 | 0 | 18.000 | 20 | 0.000 |
| ISO_NE | 3 | one_sided_w90 | 31 | 0 | NA | 19 | 0.000 |
| NYISO | 6 | two_sided_w90 | 30 | 4 | 15.000 | 21 | 0.000 |
| NYISO | 6 | one_sided_w90 | 30 | 4 | NA | 20 | 0.000 |

| dataset | method | metric | design | block_days | observed | resample_p025 | resample_p975 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ISO_NE | two_sided_w90 | paired_annual_cost_difference | annual_circular | 7 | -7.000 | -29.000 | 6.000 |
| ISO_NE | two_sided_w90 | daily_joint_coverage | annual_circular | 7 | 0.849 | 0.797 | 0.896 |
| ISO_NE | two_sided_w90 | paired_annual_cost_difference | annual_non_circular | 7 | -7.000 | -29.000 | 6.000 |
| ISO_NE | two_sided_w90 | daily_joint_coverage | annual_non_circular | 7 | 0.849 | 0.797 | 0.899 |
| ISO_NE | two_sided_w90 | paired_annual_cost_difference | quarter_stratified_non_circular | 7 | -7.000 | -29.000 | 6.000 |
| ISO_NE | two_sided_w90 | daily_joint_coverage | quarter_stratified_non_circular | 7 | 0.849 | 0.805 | 0.901 |
| ISO_NE | one_sided_w90 | paired_annual_cost_difference | annual_circular | 7 | -8.000 | -29.000 | 5.000 |
| ISO_NE | one_sided_w90 | paired_annual_cost_difference | annual_non_circular | 7 | -8.000 | -30.000 | 4.000 |
| ISO_NE | one_sided_w90 | paired_annual_cost_difference | quarter_stratified_non_circular | 7 | -8.000 | -30.000 | 5.000 |
| NYISO | two_sided_w90 | paired_annual_cost_difference | annual_circular | 7 | 11.000 | 3.000 | 21.000 |
| NYISO | two_sided_w90 | daily_joint_coverage | annual_circular | 7 | 0.844 | 0.789 | 0.893 |
| NYISO | two_sided_w90 | paired_annual_cost_difference | annual_non_circular | 7 | 11.000 | 3.000 | 21.000 |
| NYISO | two_sided_w90 | daily_joint_coverage | annual_non_circular | 7 | 0.844 | 0.789 | 0.896 |
| NYISO | two_sided_w90 | paired_annual_cost_difference | quarter_stratified_non_circular | 7 | 11.000 | 4.000 | 18.000 |
| NYISO | two_sided_w90 | daily_joint_coverage | quarter_stratified_non_circular | 7 | 0.844 | 0.795 | 0.896 |
| NYISO | one_sided_w90 | paired_annual_cost_difference | annual_circular | 7 | 9.000 | 2.000 | 18.000 |
| NYISO | one_sided_w90 | paired_annual_cost_difference | annual_non_circular | 7 | 9.000 | 2.000 | 18.000 |
| NYISO | one_sided_w90 | paired_annual_cost_difference | quarter_stratified_non_circular | 7 | 9.000 | 2.000 | 15.000 |

Daily error/loss dependence diagnostics used to inspect the block-length choice:

| dataset | series | lag_days | autocorrelation |
| --- | --- | --- | --- |
| ISO_NE | paired_daily_loss | 1 | -0.001 |
| ISO_NE | paired_daily_loss | 7 | -0.001 |
| ISO_NE | paired_daily_loss | 14 | -0.001 |
| ISO_NE | paired_daily_loss | 28 | -0.001 |
| ISO_NE | joint_failure | 1 | 0.169 |
| ISO_NE | joint_failure | 7 | 0.154 |
| ISO_NE | joint_failure | 14 | 0.071 |
| ISO_NE | joint_failure | 28 | 0.057 |
| NYISO | paired_daily_loss | 1 | 0.344 |
| NYISO | paired_daily_loss | 7 | 0.156 |
| NYISO | paired_daily_loss | 14 | 0.061 |
| NYISO | paired_daily_loss | 28 | 0.060 |
| NYISO | joint_failure | 1 | 0.235 |
| NYISO | joint_failure | 7 | 0.145 |
| NYISO | joint_failure | 14 | 0.063 |
| NYISO | joint_failure | 28 | -0.033 |

Leave-one-peak-episode-out diagnostic for episodes with nonzero paired cost:

| dataset | method | start | end | event_days | episode_paired_cost | annual_paired_cost | annual_difference_without_episode |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ISO_NE | two_sided_w90 | 2025-07-28 00:00:00 | 2025-07-30 00:00:00 | 3 | -10.000 | -7.000 | 3.000 |
| ISO_NE | one_sided_w90 | 2025-07-28 00:00:00 | 2025-07-30 00:00:00 | 3 | -10.000 | -8.000 | 2.000 |

The observed 2025 annual total is a fixed-calendar descriptive quantity. The resampling
ranges describe hypothetical reweighted day sequences under each stated block design,
conditional on the archived forecasts and validation-selected rules.

The resampling ranges condition on the stored predictions and selected rules. Annual circular
blocks wrap across December/January; the alternatives do not. None is presented as a
confidence interval for a fixed realized year or an assured future-year population.

## NYISO target-definition audit

| test_days | peak_events | irregular_days | irregular_peak_days | interpolated_days | interpolated_peak_days | dst_adjusted_days | dst_adjusted_peak_days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 365 | 17 | 317 | 17 | 0 | 0 | 2 | 0 |

Irregular timestamps are distinct local timestamp/time-zone pairs appearing off the five-minute
grid. Equal-observation hourly averaging is the evaluated target, not a duration-weighted mean.

## Tie-breaking audit

7 of 50 point-rule operating settings select a
different delta if the manuscript's smaller-parameter rule replaces the implemented
absolute-parameter-first rule. Primary results still use the released implementation.

All implemented second-round quantile ranks are finite and at most their calibration counts.
Projected ACI state boundaries and per-method rank ranges are in E29/quantile_rank_audit.csv.

## Machine-readable outputs

E24/input_hashes.csv, daily_trace_gate.csv; E25/{dataset}/window_and_aci_bands.parquet,
E25/window_decision_matrix.csv, all_decision_daily.parquet, primary_decision_daily.parquet;
E26/quality_metrics.csv and E26/{dataset}/direct_peak_daily.parquet;
E28/monthly_diagnostics.csv, resampling_sensitivity.csv, event_episode_loo.csv,
daily_error_dependence.csv; E29/nyiso_target_audit.csv, tie_break_audit.csv,
quantile_rank_audit.csv.
