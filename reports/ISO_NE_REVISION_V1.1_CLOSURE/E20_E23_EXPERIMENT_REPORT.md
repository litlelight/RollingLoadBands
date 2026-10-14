# ISO_NE_REVISION_V1.1_CLOSURE E20–E23 实验报告

> **Interpretation update (current main-branch documentation; original archived values unchanged):** Historical headings such as `95% CI` or probability-like resampling columns are superseded by the current manuscript/Supplement interpretation as *conditional descriptive 2.5th--97.5th resampling percentile ranges*, not confidence intervals for fixed observed annual costs or future-year guarantees. See [`LEGACY_INTERPRETATION_NOTICE.md`](../../LEGACY_INTERPRETATION_NOTICE.md).

本文件只记录新增实验结果，不包含论文改写、审稿回复或投稿材料。所有实验沿用已通过 E2/E3 门禁的 OOS forecast store。

## E20 Coverage moving-block bootstrap

| method | metric | observed_coverage_pct | block_length_days | ci_lower_pct | ci_upper_pct | bootstrap_fraction_below_nominal |
| --- | --- | --- | --- | --- | --- | --- |
| standardized_max_90 | daily_simultaneous_coverage | 84.932 | 7 | 79.993 | 89.863 | 0.981 |
| standardized_max_90 | daily_upper_coverage | 89.041 | 7 | 84.110 | 93.425 | 0.619 |
| one_sided_max_90 | daily_upper_coverage | 85.205 | 7 | 79.452 | 90.685 | 0.958 |
| bonferroni_365 | daily_simultaneous_coverage | 96.986 | 7 | 94.247 | 99.178 | 0.000 |
| disjoint_standardized_max_90 | daily_simultaneous_coverage | 84.932 | 7 | 79.726 | 89.863 | 0.977 |

完整文件同时包含 ISO-NE 2024/2025/2026H1 和 NYISO 2024/2025，以及 block=3/7/14/28 天。

## E21 Band robustness

| method | daily_simultaneous_coverage_pct | daily_upper_coverage_pct | peak_hour_coverage_pct | peak_hour_upper_coverage_pct | mean_upper_margin_mw | mean_width_mw | interval_score_mw |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bonferroni_365 | 96.986 | 97.808 | 100.000 | 100.000 | 3545.272 | 7090.543 | 7160.070 |
| disjoint_standardized_max_90 | 84.932 | 89.315 | 96.164 | 97.534 | 2771.753 | 5543.506 | 5828.175 |
| one_sided_max_90 | NA | 85.205 | NA | 96.712 | 2130.807 | NA | NA |
| rolling_marginal_90 | 61.096 | 76.438 | 87.123 | 89.863 | 1348.839 | 2697.678 | 3964.684 |
| rolling_raw_max_90 | 89.589 | 93.973 | 96.438 | 96.986 | 2605.676 | 5211.352 | 5611.933 |
| standardized_max_90 | 84.932 | 89.041 | 97.808 | 98.082 | 2471.865 | 4943.730 | 5189.695 |
| static_standardized_previous_year | 84.384 | 89.315 | 94.795 | 96.438 | 2013.930 | 4027.859 | 4645.779 |

另行保存 calibration-window 30/60/90/180/270/365 天与 scale-floor 25/50/100 MW 的完整逐日 bands 和指标。

### 2024 calibration-window selection audit

| calibration_window_days | daily_simultaneous_coverage_pct | absolute_daily_coverage_gap_pp | mean_width_mw | interval_score_mw | rank_coverage_closeness | rank_mean_width | rank_interval_score | pareto_dominated_on_three_reported_metrics | originally_frozen_window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 88.251 | 1.749 | 4560.955 | 4659.008 | 1 | 6 | 6 | False | False |
| 60 | 84.426 | 5.574 | 4142.532 | 4273.744 | 6 | 4 | 3 | True | False |
| 90 | 85.792 | 4.208 | 4093.812 | 4234.960 | 4 | 3 | 2 | True | True |
| 180 | 86.339 | 3.661 | 4149.088 | 4381.353 | 3 | 5 | 5 | True | False |
| 270 | 85.792 | 4.208 | 3958.897 | 4300.701 | 4 | 2 | 4 | True | False |
| 365 | 87.158 | 2.842 | 3824.670 | 4104.495 | 2 | 1 | 1 | False | False |

归档实验记录无法恢复当时的单一 objective/selection criterion；90 天仅按原冻结设置保留，不把 2024 或 2025 的事后最优值重新定义为主配置。

## E22 Development-period sensitivity

### Ridge α × training window（2020–2023 only）

| configuration | ridge_alpha | training_window_years | frozen_primary | mae_mw | rmse_mw | peak_mae_mw |
| --- | --- | --- | --- | --- | --- | --- |
| ridge_a1_w7y | 1.000 | 7 | False | 523.194 | 796.573 | 721.880 |
| ridge_a1_w5y | 1.000 | 5 | False | 523.423 | 799.107 | 720.486 |
| ridge_a10_w5y | 10.000 | 5 | False | 524.357 | 795.984 | 709.508 |
| ridge_a10_w7y | 10.000 | 7 | False | 525.913 | 796.221 | 710.958 |
| ridge_a100_w7y | 100.000 | 7 | False | 536.835 | 803.447 | 702.807 |
| ridge_a100_w5y | 100.000 | 5 | True | 537.130 | 803.697 | 700.135 |
| ridge_a10_w3y | 10.000 | 3 | False | 539.934 | 819.199 | 737.502 |
| ridge_a100_w3y | 100.000 | 3 | False | 549.429 | 819.289 | 710.123 |
| ridge_a1_w3y | 1.000 | 3 | False | 553.525 | 842.614 | 768.763 |
| ridge_a10_w2y | 10.000 | 2 | False | 556.260 | 846.905 | 765.889 |
| ridge_a100_w2y | 100.000 | 2 | False | 566.012 | 841.963 | 727.068 |
| ridge_a1_w2y | 1.000 | 2 | False | 584.148 | 887.228 | 829.744 |
| ridge_a1000_w7y | 1000.000 | 7 | False | 585.816 | 852.581 | 711.566 |
| ridge_a1000_w5y | 1000.000 | 5 | False | 599.824 | 868.204 | 726.518 |
| ridge_a1000_w3y | 1000.000 | 3 | False | 624.412 | 897.800 | 761.734 |
| ridge_a1000_w2y | 1000.000 | 2 | False | 653.845 | 934.297 | 801.917 |

### Calibration window（2020–2023 only）

| calibration_window_days | frozen_primary | daily_simultaneous_coverage_pct | mean_width_mw | interval_score_mw | absolute_daily_coverage_error_pp |
| --- | --- | --- | --- | --- | --- |
| 30 | False | 87.885 | 4091.012 | 4226.697 | 2.115 |
| 60 | False | 88.022 | 3852.623 | 4025.445 | 1.978 |
| 90 | True | 87.817 | 3832.047 | 4049.440 | 2.183 |
| 180 | False | 88.706 | 3849.975 | 4134.903 | 1.294 |
| 270 | False | 89.596 | 3861.526 | 4221.444 | 0.404 |
| 365 | False | 90.554 | 3822.648 | 4085.436 | 0.554 |

这些表是开发期敏感性，不使用 2025 结果重新选择主配置。

## E23 2025 peak-event audit

共 17 个事件日，按 pre-validation frozen historical Q95 threshold 定义。逐事件 alert、decision score 和 daily cost 均保存在 CSV。
