# Expected experimental checkpoints

These values are compact checkpoints for confirming that a rerun followed the frozen pipeline. Full precision is stored in the CSV outputs.

## ISO-NE 2025 point forecasts

| Model | MAE (MW) | RMSE (MW) | Peak MAE (MW) |
| --- | ---: | ---: | ---: |
| Ridge | 611.656 | 945.908 | 721.494 |
| Previous day (D-1) | 970.681 | 1404.333 | 961.496 |
| Previous week (D-7) | 1436.523 | 2075.543 | 1532.145 |
| XGBoost selected candidate | 644.491 | 984.869 | 761.658 |
| TCN five-seed ensemble | 876.150 | 1234.982 | 967.332 |

## ISO-NE 2025 uncertainty and decisions

- Standardized-max 90-day daily simultaneous coverage: **84.932%**.
- Standardized-max 90-day daily upper coverage: **89.041%**.
- One-sided 90-day daily upper coverage: **85.205%**.
- One-sided mean upper margin: **2130.807 MW**.
- Historical-Q95, cost ratio 10, one-sided conformal decision: TP 17, FP 16, FN 0, total cost 16.
- Historical-Q95, cost ratio 10, separately tuned point decision: TP 16, FP 14, FN 1, total cost 24.

## NYISO 2025 external validation

- Ridge MAE: **610.657 MW**.
- Standardized-max 90-day daily upper coverage: **92.329%**.
- One-sided 90-day daily upper coverage: **89.589%**.

## Closure audits

- E20 contains 100 coverage moving-block-bootstrap rows.
- E23 contains 17 ISO-NE peak-event rows.
- The originally frozen 90-day calibration window is explicitly marked as Pareto-dominated by the 365-day candidate on the three reported 2024 metrics; it is retained rather than changed post hoc.

## Second-round E24--E29 checkpoints

- `results/ROUND2_REVISION_V1/RUN_STATUS.json` reports `PASS`.
- E24 checks nine frozen inputs and reconciles all 1,050 decision-matrix rows with 365 daily decisions each (383,250 daily records). Frozen file hashes must remain unchanged.
- At historical Q95 and missed-event cost ratio 10, the inherited 90-day ISO-NE two-sided/one-sided/tuned-point 2025 costs are 17/16/24; NYISO costs are 22/20/11.
- The 2024-selected projected-ACI step size is 0.02 in both systems. Its 2025 joint coverage is 89.589% (ISO-NE) and 90.137% (NYISO); these observed values do not constitute a coverage guarantee.
- Direct daily-maximum calibration produces a scalar peak bound, not a simultaneous 24-position band. Its 90-day 2025 cost is 20 (ISO-NE) and 31 (NYISO) at the same operating point.
- The 2025 NYISO target audit reports 317 irregular-timestamp days, all 17 Q95 peak-event days affected, zero 2025 interpolated days, and two DST-adjusted days, neither a peak event.
- The primary scientific outputs reproduced byte-for-byte in the staged v1.2.0 run. Only E24 path metadata and `RUN_STATUS.json` differ from the original local run; the new report records the new runtime.
