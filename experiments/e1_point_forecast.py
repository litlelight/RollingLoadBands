from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "work" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


EVALUATION_START = pd.Timestamp("2025-01-01")
EVALUATION_END = pd.Timestamp("2025-12-31")
RIDGE_ALPHA = 100.0
TRAINING_YEARS = 5
MW_TOLERANCE = 0.1
MAPE_TOLERANCE = 0.01

REPORTED_TARGETS: dict[str, dict[str, float]] = {
    "ridge_full": {"mae_mw": 610.91, "rmse_mw": 945.46, "mape_pct": 4.45, "peak_mae_mw": 721.62},
    "ridge_no_calendar": {"mae_mw": 627.36, "mape_pct": 4.55, "peak_mae_mw": 763.87},
    "ridge_no_d1_profile": {"mae_mw": 707.33, "mape_pct": 5.14, "peak_mae_mw": 818.80},
    "d1_persistence": {"mae_mw": 970.75, "rmse_mw": 1404.36, "mape_pct": 7.12, "peak_mae_mw": 961.50},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the ISO-NE 2025 point-forecast experiment.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs" / "01_data_audit",
        help="Directory containing E0 Parquet files and feature manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "02_e1_point_forecast",
        help="Directory for E1 artifacts.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error / actual)) * 100),
        "peak_mae_mw": float(np.mean(np.abs(actual.max(axis=1) - predicted.max(axis=1)))),
    }


def fit_monthly_ridge(
    features: pd.DataFrame,
    feature_names: list[str],
    target_names: list[str],
    model_name: str,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    dates = pd.to_datetime(features["date"])
    x = features[feature_names].to_numpy(dtype=float)
    y = features[target_names].to_numpy(dtype=float)
    prediction = np.full_like(y, np.nan)
    audit_rows: list[dict[str, object]] = []

    for month_start in pd.date_range(EVALUATION_START, EVALUATION_END, freq="MS"):
        planned_training_start = month_start - pd.DateOffset(years=TRAINING_YEARS)
        training_end = month_start - pd.Timedelta(days=1)
        prediction_end = month_start + pd.offsets.MonthEnd(0)
        train_mask = dates.between(planned_training_start, training_end).to_numpy()
        test_mask = dates.between(month_start, prediction_end).to_numpy()

        train_dates = dates[train_mask]
        test_dates = dates[test_mask]
        if not len(train_dates) or not len(test_dates):
            raise AssertionError(f"Empty train or test window for {month_start:%Y-%m}")
        if train_dates.max() >= test_dates.min():
            raise AssertionError(f"Temporal leakage in {month_start:%Y-%m}")
        if not np.isfinite(x[train_mask]).all() or not np.isfinite(y[train_mask]).all():
            raise AssertionError(f"Non-finite training data in {month_start:%Y-%m}")

        scaler = StandardScaler()
        model = Ridge(alpha=RIDGE_ALPHA)
        x_train = scaler.fit_transform(x[train_mask])
        model.fit(x_train, y[train_mask])
        prediction[test_mask] = model.predict(scaler.transform(x[test_mask]))

        audit_rows.append(
            {
                "model": model_name,
                "fit_month": month_start.strftime("%Y-%m"),
                "planned_training_start": planned_training_start.date().isoformat(),
                "actual_training_start": train_dates.min().date().isoformat(),
                "training_end": train_dates.max().date().isoformat(),
                "scaler_fit_start": train_dates.min().date().isoformat(),
                "scaler_fit_end": train_dates.max().date().isoformat(),
                "prediction_start": test_dates.min().date().isoformat(),
                "prediction_end": test_dates.max().date().isoformat(),
                "training_days": int(train_mask.sum()),
                "prediction_days": int(test_mask.sum()),
                "feature_count": len(feature_names),
                "ridge_alpha": RIDGE_ALPHA,
                "no_temporal_leakage": bool(train_dates.max() < test_dates.min()),
            }
        )

    evaluation_mask = dates.between(EVALUATION_START, EVALUATION_END).to_numpy()
    if not np.isfinite(prediction[evaluation_mask]).all():
        raise AssertionError("At least one 2025 Ridge prediction is missing or non-finite")
    return prediction, audit_rows


def daily_metrics(
    dates: pd.Series,
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    evaluation_mask: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, prediction in predictions.items():
        for date, actual_day, predicted_day in zip(
            dates[evaluation_mask], actual[evaluation_mask], prediction[evaluation_mask]
        ):
            error = actual_day - predicted_day
            rows.append(
                {
                    "date": pd.Timestamp(date).date().isoformat(),
                    "model": model_name,
                    "mae_mw": float(np.mean(np.abs(error))),
                    "rmse_mw": float(np.sqrt(np.mean(error**2))),
                    "mape_pct": float(np.mean(np.abs(error / actual_day)) * 100),
                    "actual_peak_mw": float(actual_day.max()),
                    "predicted_peak_mw": float(predicted_day.max()),
                    "peak_absolute_error_mw": float(abs(actual_day.max() - predicted_day.max())),
                }
            )
    return pd.DataFrame(rows)


def metric_table(
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    feature_counts: dict[str, int],
    evaluation_mask: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, prediction in predictions.items():
        values = point_metrics(actual[evaluation_mask], prediction[evaluation_mask])
        row: dict[str, object] = {
            "model": model_name,
            "evaluation_start": EVALUATION_START.date().isoformat(),
            "evaluation_end": EVALUATION_END.date().isoformat(),
            "evaluation_days": int(evaluation_mask.sum()),
            "hourly_positions": int(evaluation_mask.sum() * actual.shape[1]),
            "feature_count": feature_counts[model_name],
            **values,
        }
        target_checks = []
        for metric_name in ("mae_mw", "rmse_mw", "mape_pct", "peak_mae_mw"):
            target = REPORTED_TARGETS.get(model_name, {}).get(metric_name)
            difference = abs(values[metric_name] - target) if target is not None else np.nan
            tolerance = MAPE_TOLERANCE if metric_name == "mape_pct" else MW_TOLERANCE
            passed = bool(difference < tolerance) if target is not None else None
            row[f"reported_{metric_name}"] = target
            row[f"absolute_difference_{metric_name}"] = difference
            row[f"within_tolerance_{metric_name}"] = passed
            if passed is not None:
                target_checks.append(passed)
        row["all_available_anchors_within_tolerance"] = bool(all(target_checks))
        rows.append(row)
    return pd.DataFrame(rows)


def report_text(metrics: pd.DataFrame, input_hashes: dict[str, str], status: str) -> str:
    by_model = metrics.set_index("model")

    def metric_row(model: str, label: str) -> str:
        row = by_model.loc[model]
        return (
            f"| {label} | {int(row['feature_count'])} | {row['mae_mw']:.2f} | "
            f"{row['rmse_mw']:.2f} | {row['mape_pct']:.2f} | {row['peak_mae_mw']:.2f} | "
            f"{'PASS' if row['all_available_anchors_within_tolerance'] else 'FAIL'} |"
        )

    full = by_model.loc["ridge_full"]
    d1 = by_model.loc["d1_persistence"]
    return f"""# E1 ISO-NE Point-Forecast Reproduction Report

## Outcome

E1 gate status: **{status}**

The D-1 persistence benchmark reproduces the manuscript within the stated 0.1 MW tolerance, which independently validates the cleaned 2025 ISO-NE profile and the evaluation formulas. The reconstructed full Ridge model does not reproduce all manuscript anchors within tolerance, so E2 is blocked under the preregistered experiment order.

## Protocol

- Evaluation: 2025-01-01 through 2025-12-31 (365 daily 24-position profiles).
- Point model: multi-output Ridge regression with `alpha=100` and an intercept.
- Updates: one model fitted at the start of each calendar month.
- Training data: the most recent five calendar years ending the day before each update.
- Standardization: `StandardScaler` fitted only to each model's training window, then applied to that month's targets.
- Predictors: the E0 233-feature manifest, including daily-summary lags d-1, d-7, and d-28.
- All date boundaries are inclusive and every training/scaler end date precedes its prediction start date.

## Aggregate results

| Model | Features | MAE (MW) | RMSE (MW) | MAPE (%) | Peak MAE (MW) | Anchor status |
|---|---:|---:|---:|---:|---:|---|
{metric_row('ridge_full', 'Ridge, full')}
{metric_row('ridge_no_calendar', 'Ridge, no calendar')}
{metric_row('ridge_no_d1_profile', 'Ridge, no d-1 profile')}
{metric_row('d1_persistence', 'D-1 persistence')}

The manuscript's full-model targets are 610.91 MW MAE, 945.46 MW RMSE, 4.45% MAPE, and 721.62 MW peak MAE. The reconstruction gives {full['mae_mw']:.6f}, {full['rmse_mw']:.6f}, {full['mape_pct']:.6f}%, and {full['peak_mae_mw']:.6f} MW, respectively. Absolute differences are {full['absolute_difference_mae_mw']:.6f}, {full['absolute_difference_rmse_mw']:.6f}, {full['absolute_difference_mape_pct']:.6f} percentage points, and {full['absolute_difference_peak_mae_mw']:.6f} MW.

The D-1 MAE is {d1['mae_mw']:.6f} MW versus the reported 970.75 MW (absolute difference {d1['absolute_difference_mae_mw']:.6f} MW).

## Diagnosis

The feature-count evidence identifies the manuscript's otherwise unstated 21 daily-summary predictors as seven summaries at lags d-1, d-7, and d-28: this gives the reported 233 full features, 205 features without the 28 calendar variables, and 209 features after removing the 24-hour d-1 profile. The no-calendar Ridge nearly reproduces its available anchors, while the full model does not. Together with the exact D-1 reproduction, this localizes the unresolved discrepancy to the precise calendar-feature implementation or to an unreported implementation detail, rather than to ISO-NE parsing or metric calculation.

Several defensible U.S. federal-holiday conventions were checked during diagnosis (current observed federal dates, actual dates, and the pre-Juneteenth definition); none reproduced all full-model anchors. They were not selected post hoc. Exact reproduction now requires the original code, the original 28-column calendar design matrix, or a complete definition of all calendar indicators.

## Gate decision

E1 requires every full-model anchor to be within 0.1 MW (and the two-decimal MAPE within 0.01 percentage points). The full Ridge model fails that gate. **Do not start E2 from these reconstructed predictions.**

## Reproducibility artifacts

- `e1_metrics.csv`: aggregate values, reported anchors, absolute differences, and pass/fail checks.
- `e1_daily_metrics.csv`: daily errors for every model variant.
- `e1_predictions.parquet`: actual and predicted 24-position profiles for 2025.
- `e1_monthly_fit_audit.csv`: training, scaler, and prediction date boundaries for every monthly fit.
- `e1_run_config.json`: exact settings, software versions, and input hashes.

Input SHA-256 values:

- ISO_NE_features.parquet: `{input_hashes['features']}`
- ISO_NE_daily_profile.parquet: `{input_hashes['daily']}`
- feature_manifest.csv: `{input_hashes['manifest']}`
"""


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = input_dir / "ISO_NE_features.parquet"
    daily_path = input_dir / "ISO_NE_daily_profile.parquet"
    manifest_path = input_dir / "feature_manifest.csv"
    for path in (feature_path, daily_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    features = pd.read_parquet(feature_path).sort_values("date").reset_index(drop=True)
    features["date"] = pd.to_datetime(features["date"])
    manifest = pd.read_csv(manifest_path).sort_values("feature_index")
    feature_names = manifest["feature_name"].tolist()
    target_names = [f"target_load_h{hour:02d}" for hour in range(1, 25)]
    if len(feature_names) != 233:
        raise AssertionError(f"Expected 233 predictors, found {len(feature_names)}")
    if not set(feature_names + target_names + ["date"]).issubset(features.columns):
        raise AssertionError("E0 feature table does not contain all required columns")

    no_calendar = manifest.loc[
        ~manifest["category"].isin(["calendar", "calendar_one_hot"]), "feature_name"
    ].tolist()
    no_d1_profile = [name for name in feature_names if not name.startswith("load_lag1_h")]
    if len(no_calendar) != 205 or len(no_d1_profile) != 209:
        raise AssertionError(
            f"Unexpected ablation dimensions: no-calendar={len(no_calendar)}, no-d1={len(no_d1_profile)}"
        )

    model_features = {
        "ridge_full": feature_names,
        "ridge_no_calendar": no_calendar,
        "ridge_no_d1_profile": no_d1_profile,
    }
    predictions: dict[str, np.ndarray] = {}
    audit_rows: list[dict[str, object]] = []
    for model_name, selected_features in model_features.items():
        predictions[model_name], audit = fit_monthly_ridge(
            features, selected_features, target_names, model_name
        )
        audit_rows.extend(audit)

    actual = features[target_names].to_numpy(dtype=float)
    d1_columns = [f"load_lag1_h{hour:02d}" for hour in range(1, 25)]
    predictions["d1_persistence"] = features[d1_columns].to_numpy(dtype=float)
    dates = pd.to_datetime(features["date"])
    evaluation_mask = dates.between(EVALUATION_START, EVALUATION_END).to_numpy()
    if evaluation_mask.sum() != 365:
        raise AssertionError(f"Expected 365 evaluation days, found {evaluation_mask.sum()}")

    feature_counts = {name: len(columns) for name, columns in model_features.items()}
    feature_counts["d1_persistence"] = 24
    metrics = metric_table(actual, predictions, feature_counts, evaluation_mask)
    metrics.to_csv(output_dir / "e1_metrics.csv", index=False, float_format="%.10f")

    daily = daily_metrics(dates, actual, predictions, evaluation_mask)
    daily.to_csv(output_dir / "e1_daily_metrics.csv", index=False, float_format="%.10f")
    pd.DataFrame(audit_rows).to_csv(output_dir / "e1_monthly_fit_audit.csv", index=False)

    prediction_columns: dict[str, object] = {"date": dates[evaluation_mask].to_numpy()}
    for hour_index in range(24):
        prediction_columns[f"actual_h{hour_index + 1:02d}"] = actual[evaluation_mask, hour_index]
    for model_name, prediction in predictions.items():
        for hour_index in range(24):
            prediction_columns[f"{model_name}_h{hour_index + 1:02d}"] = prediction[
                evaluation_mask, hour_index
            ]
    prediction_frame = pd.DataFrame(prediction_columns)
    prediction_frame.to_parquet(
        output_dir / "e1_predictions.parquet", index=False, engine="pyarrow", compression="zstd"
    )

    full_row = metrics.loc[metrics["model"] == "ridge_full"].iloc[0]
    status = "PASS" if bool(full_row["all_available_anchors_within_tolerance"]) else "FAIL - E2 BLOCKED"
    input_hashes = {
        "features": sha256(feature_path),
        "daily": sha256(daily_path),
        "manifest": sha256(manifest_path),
    }
    config = {
        "experiment": "E1 ISO-NE point-forecast reproduction",
        "status": status,
        "evaluation_start": EVALUATION_START.date().isoformat(),
        "evaluation_end": EVALUATION_END.date().isoformat(),
        "training_years": TRAINING_YEARS,
        "update_frequency": "calendar month start",
        "ridge_alpha": RIDGE_ALPHA,
        "scaler": "StandardScaler fitted separately on each monthly training window",
        "mw_tolerance_strictly_less_than": MW_TOLERANCE,
        "mape_percentage_point_tolerance_strictly_less_than": MAPE_TOLERANCE,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "input_sha256": input_hashes,
    }
    (output_dir / "e1_run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = report_text(metrics, input_hashes, status)
    (output_dir / "e1_reproduction_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
