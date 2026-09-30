from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "work" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


LOAD_HOURS = 24
LOAD_COLUMNS = [f"load_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]
TARGET_COLUMNS = [f"target_load_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]
TRUE_COLUMNS = [f"y_true_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]
PRED_COLUMNS = [f"y_pred_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]
RESIDUAL_COLUMNS = [f"residual_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{now_text()}] {message}", flush=True)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def quantile_higher(values: np.ndarray, rank: int) -> np.ndarray:
    if rank < 1 or rank > values.shape[0]:
        raise ValueError(f"Order statistic rank {rank} is outside 1..{values.shape[0]}")
    return np.sort(values, axis=0)[rank - 1]


def finite_sample_rank(n: int, alpha: float, cap: bool = False) -> int:
    rank = int(math.ceil((n + 1) * (1 - alpha)))
    if cap:
        rank = min(n, rank)
    return rank


def point_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error / actual)) * 100),
        "peak_mae_mw": float(np.mean(np.abs(actual.max(axis=1) - predicted.max(axis=1)))),
    }


def period_label(date: pd.Timestamp) -> str:
    if date.year == 2026:
        return "2026H1"
    return str(date.year)


def evaluation_periods() -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    periods = [
        (str(year), pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
        for year in range(2020, 2026)
    ]
    periods.append(("2026H1", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-30")))
    return periods


def hardware_snapshot() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    result: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "total_ram_gb": round(memory.total / 1024**3, 3),
        "available_ram_gb_at_start": round(memory.available / 1024**3, 3),
    }
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        result["nvidia_smi"] = query.stdout.strip()
    except Exception as exc:
        result["nvidia_smi"] = f"unavailable: {exc}"
    return result


def software_snapshot() -> dict[str, str]:
    packages = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    for package in ("pyarrow", "psutil", "yaml", "xgboost", "torch", "matplotlib", "holidays"):
        try:
            module = __import__(package)
            packages[package] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            packages[package] = f"unavailable: {exc}"
    return packages


@dataclass
class StageTimer:
    name: str
    started: float = 0.0
    start_memory: int = 0

    def __enter__(self) -> "StageTimer":
        self.started = time.perf_counter()
        self.start_memory = psutil.Process().memory_info().rss
        log(f"START {self.name}")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        elapsed = time.perf_counter() - self.started
        memory = psutil.Process().memory_info().rss
        status = "FAIL" if exc_type else "PASS"
        log(f"END {self.name}: {status}, {elapsed:.2f} seconds, process RSS {memory / 1024**3:.3f} GB")


def freeze_check(config: dict[str, Any], e0_dir: Path) -> pd.DataFrame:
    manifest_path = e0_dir / "feature_manifest.csv"
    features_path = e0_dir / "ISO_NE_features.parquet"
    daily_path = e0_dir / "ISO_NE_daily_profile.parquet"
    manifest = pd.read_csv(manifest_path)
    summary = manifest.loc[manifest["category"] == "lagged_daily_summary"]
    actual_lags = sorted(int(value.removeprefix("d-")) for value in summary["source_date_rule"].unique())
    expected = config["freeze"]
    checks = [
        ("feature_count", len(manifest), int(expected["feature_count"])),
        ("summary_lags", actual_lags, list(expected["summary_lags"])),
        ("iso_features_sha256", file_sha256(features_path), expected["iso_features_sha256"]),
        ("iso_daily_sha256", file_sha256(daily_path), expected["iso_daily_sha256"]),
        ("feature_manifest_sha256", file_sha256(manifest_path), expected["feature_manifest_sha256"]),
    ]
    rows = []
    for name, actual, expected_value in checks:
        rows.append(
            {
                "check": name,
                "actual": json.dumps(actual) if isinstance(actual, list) else str(actual),
                "expected": json.dumps(expected_value) if isinstance(expected_value, list) else str(expected_value),
                "status": "PASS" if actual == expected_value else "FAIL",
            }
        )
    result = pd.DataFrame(rows)
    if not (result["status"] == "PASS").all():
        raise AssertionError("Frozen-pipeline input check failed")
    return result


def load_feature_inputs(e0_dir: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix = "ISO_NE" if dataset == "ISO_NE" else "NYISO"
    features = pd.read_parquet(e0_dir / f"{prefix}_features.parquet").sort_values("date").reset_index(drop=True)
    features["date"] = pd.to_datetime(features["date"])
    daily = pd.read_parquet(e0_dir / f"{prefix}_daily_profile.parquet").sort_values("date").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    manifest = pd.read_csv(e0_dir / "feature_manifest.csv").sort_values("feature_index")
    return features, daily, manifest


def generate_oos_ridge(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    alpha: float,
    training_years: int,
    dataset: str,
    cpu_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(features["date"])
    feature_names = manifest["feature_name"].tolist()
    if len(feature_names) != 233:
        raise AssertionError(f"Expected 233 features; found {len(feature_names)}")
    x = features[feature_names].to_numpy(dtype=np.float64)
    y = features[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    prediction = np.full_like(y, np.nan)
    metadata: dict[int, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    for month_start in pd.date_range(start_ts.replace(day=1), end_ts, freq="MS"):
        planned_start = month_start - pd.DateOffset(years=training_years)
        training_end = month_start - pd.Timedelta(days=1)
        month_end = min(month_start + pd.offsets.MonthEnd(0), end_ts)
        train_mask = dates.between(planned_start, training_end).to_numpy()
        test_mask = dates.between(max(month_start, start_ts), month_end).to_numpy()
        if not test_mask.any():
            continue
        if not train_mask.any():
            raise AssertionError(f"No training rows for {dataset} {month_start:%Y-%m}")
        train_dates = dates[train_mask]
        test_dates = dates[test_mask]
        if train_dates.max() >= test_dates.min():
            raise AssertionError(f"Training overlap for {dataset} {month_start:%Y-%m}")
        scaler = StandardScaler()
        model = Ridge(alpha=float(alpha))
        with threadpool_limits(limits=max(1, int(cpu_threads))):
            model.fit(scaler.fit_transform(x[train_mask]), y[train_mask])
            prediction[test_mask] = model.predict(scaler.transform(x[test_mask]))
        model_id = f"{dataset}_ridge_{month_start:%Y_%m}"
        for index in np.flatnonzero(test_mask):
            metadata[int(index)] = {
                "model_id": model_id,
                "model_fit_date": training_end,
                "training_start": train_dates.min(),
                "training_end": train_dates.max(),
                "scaler_fit_start": train_dates.min(),
                "scaler_fit_end": train_dates.max(),
            }
        audit_rows.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "month": month_start.strftime("%Y-%m"),
                "planned_training_start": planned_start,
                "training_start": train_dates.min(),
                "training_end": train_dates.max(),
                "scaler_fit_start": train_dates.min(),
                "scaler_fit_end": train_dates.max(),
                "prediction_start": test_dates.min(),
                "prediction_end": test_dates.max(),
                "training_rows": int(train_mask.sum()),
                "prediction_rows": int(test_mask.sum()),
                "feature_count": len(feature_names),
                "alpha": float(alpha),
                "no_temporal_leakage": bool(train_dates.max() < test_dates.min()),
            }
        )
        log(f"{dataset} Ridge {month_start:%Y-%m}: trained {int(train_mask.sum())}, predicted {int(test_mask.sum())}")

    selection = dates.between(start_ts, end_ts).to_numpy()
    selected_indices = np.flatnonzero(selection)
    if not np.isfinite(prediction[selection]).all():
        raise AssertionError(f"Missing OOS predictions for {dataset}")
    actual = y[selection]
    predicted = prediction[selection]
    residual = actual - predicted
    records: dict[str, Any] = {
        "target_date": dates[selection].to_numpy(),
        "dataset": dataset,
        "model_id": [metadata[int(index)]["model_id"] for index in selected_indices],
        "model_fit_date": [metadata[int(index)]["model_fit_date"] for index in selected_indices],
        "training_start": [metadata[int(index)]["training_start"] for index in selected_indices],
        "training_end": [metadata[int(index)]["training_end"] for index in selected_indices],
        "scaler_fit_start": [metadata[int(index)]["scaler_fit_start"] for index in selected_indices],
        "scaler_fit_end": [metadata[int(index)]["scaler_fit_end"] for index in selected_indices],
        "residual_available_date": dates[selection].to_numpy(),
    }
    for hour_index in range(LOAD_HOURS):
        records[TRUE_COLUMNS[hour_index]] = actual[:, hour_index]
        records[PRED_COLUMNS[hour_index]] = predicted[:, hour_index]
        records[RESIDUAL_COLUMNS[hour_index]] = residual[:, hour_index]
    store = pd.DataFrame(records)
    return store, pd.DataFrame(audit_rows)


def audit_oos_and_calibration(
    store: pd.DataFrame,
    audit: pd.DataFrame,
    evaluation_start: str | pd.Timestamp,
    evaluation_end: str | pd.Timestamp,
    methods: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    data = store.sort_values("target_date").reset_index(drop=True).copy()
    for column in (
        "target_date",
        "model_fit_date",
        "training_start",
        "training_end",
        "scaler_fit_start",
        "scaler_fit_end",
        "residual_available_date",
    ):
        data[column] = pd.to_datetime(data[column])
    violations: list[dict[str, Any]] = []
    duplicate_dates = data["target_date"].duplicated(keep=False)
    for date in data.loc[duplicate_dates, "target_date"]:
        violations.append({"target_date": date, "method": "OOS", "violation": "duplicate target date"})
    date_checks = {
        "training_end_not_before_target": data["training_end"] >= data["target_date"],
        "scaler_end_not_before_target": data["scaler_fit_end"] >= data["target_date"],
        "model_fit_not_before_target": data["model_fit_date"] >= data["target_date"],
        "residual_availability_mismatch": data["residual_available_date"] != data["target_date"],
    }
    for reason, mask in date_checks.items():
        for date in data.loc[mask, "target_date"]:
            violations.append({"target_date": date, "method": "OOS", "violation": reason})
    recomputed = data[TRUE_COLUMNS].to_numpy(float) - data[PRED_COLUMNS].to_numpy(float)
    residual_error = np.max(np.abs(recomputed - data[RESIDUAL_COLUMNS].to_numpy(float)), axis=1)
    for date, error in zip(data.loc[residual_error > 1e-9, "target_date"], residual_error[residual_error > 1e-9]):
        violations.append({"target_date": date, "method": "OOS", "violation": f"residual mismatch {error}"})

    date_set = set(data["target_date"])
    eval_dates = pd.date_range(pd.Timestamp(evaluation_start), pd.Timestamp(evaluation_end), freq="D")
    summary_rows: list[dict[str, Any]] = []
    for method, history_days in methods.items():
        method_violations = 0
        for target in eval_dates:
            required = [target - pd.Timedelta(days=value) for value in range(1, history_days + 1)]
            missing = [date for date in required if date not in date_set]
            if missing:
                method_violations += 1
                violations.append(
                    {
                        "target_date": target,
                        "method": method,
                        "violation": f"missing {len(missing)} required prior OOS residual dates",
                    }
                )
        summary_rows.append(
            {
                "method": method,
                "evaluation_start": pd.Timestamp(evaluation_start),
                "evaluation_end": pd.Timestamp(evaluation_end),
                "required_history_days": history_days,
                "evaluated_target_days": len(eval_dates),
                "violating_target_days": method_violations,
                "status": "PASS" if method_violations == 0 else "FAIL",
            }
        )
    monthly_pass = bool(audit["no_temporal_leakage"].all())
    summary_rows.append(
        {
            "method": "monthly_model_and_scaler",
            "evaluation_start": pd.to_datetime(audit["prediction_start"]).min(),
            "evaluation_end": pd.to_datetime(audit["prediction_end"]).max(),
            "required_history_days": np.nan,
            "evaluated_target_days": int(audit["prediction_rows"].sum()),
            "violating_target_days": int((~audit["no_temporal_leakage"].astype(bool)).sum()),
            "status": "PASS" if monthly_pass else "FAIL",
        }
    )
    summary = pd.DataFrame(summary_rows)
    violations_frame = pd.DataFrame(violations, columns=["target_date", "method", "violation"])
    gate = len(violations_frame) == 0 and (summary["status"] == "PASS").all()
    return summary, violations_frame, bool(gate)


def _window_indices(date_to_index: dict[pd.Timestamp, int], target: pd.Timestamp, start_lag: int, end_lag: int) -> list[int]:
    dates = [target - pd.Timedelta(days=lag) for lag in range(start_lag, end_lag - 1, -1)]
    missing = [date for date in dates if date not in date_to_index]
    if missing:
        raise KeyError(f"Missing {len(missing)} dates for {target:%Y-%m-%d}: {missing[:3]}")
    return [date_to_index[date] for date in dates]


def build_conformal_store(store: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data = store.sort_values("target_date").reset_index(drop=True).copy()
    data["target_date"] = pd.to_datetime(data["target_date"])
    residual = data[RESIDUAL_COLUMNS].to_numpy(float)
    predicted = data[PRED_COLUMNS].to_numpy(float)
    date_to_index = {pd.Timestamp(date): index for index, date in enumerate(data["target_date"])}
    alpha = float(config["alpha"])
    primary_n = int(config["primary_window_days"])
    bonf_n = int(config["bonferroni_window_days"])
    scale_n = int(config["disjoint_scale_days"])
    score_n = int(config["disjoint_score_days"])
    floor = float(config["scale_floor_mw"])
    rows: list[dict[str, Any]] = []

    for row_index, target in enumerate(data["target_date"]):
        method_specs: list[tuple[str, int]] = [
            ("standardized_max_90", primary_n),
            ("one_sided_max_90", primary_n),
            ("disjoint_standardized_max_90", scale_n + score_n),
            ("bonferroni_365", bonf_n),
        ]
        for method, required_history in method_specs:
            if target - pd.Timedelta(days=required_history) not in date_to_index:
                continue
            if method in ("standardized_max_90", "one_sided_max_90"):
                indexes = _window_indices(date_to_index, target, primary_n, 1)
                window = residual[indexes]
                sigma = np.maximum(floor, np.median(np.abs(window), axis=0))
                if method == "standardized_max_90":
                    scores = np.max(np.abs(window) / sigma, axis=1)
                else:
                    scores = np.max(window / sigma, axis=1)
                rank = finite_sample_rank(primary_n, alpha, cap=True)
                q = float(quantile_higher(scores[:, None], rank)[0])
                margin = q * sigma
                lower = predicted[row_index] - margin if method == "standardized_max_90" else np.full(LOAD_HOURS, np.nan)
                upper = predicted[row_index] + margin
                scale_start = target - pd.Timedelta(days=primary_n)
                scale_end = target - pd.Timedelta(days=1)
                score_start = scale_start
                score_end = scale_end
                q_hours = np.full(LOAD_HOURS, np.nan)
            elif method == "disjoint_standardized_max_90":
                scale_indexes = _window_indices(date_to_index, target, scale_n + score_n, score_n + 1)
                score_indexes = _window_indices(date_to_index, target, score_n, 1)
                scale_window = residual[scale_indexes]
                score_window = residual[score_indexes]
                sigma = np.maximum(floor, np.median(np.abs(scale_window), axis=0))
                scores = np.max(np.abs(score_window) / sigma, axis=1)
                rank = finite_sample_rank(score_n, alpha, cap=True)
                q = float(quantile_higher(scores[:, None], rank)[0])
                margin = q * sigma
                lower = predicted[row_index] - margin
                upper = predicted[row_index] + margin
                scale_start = target - pd.Timedelta(days=scale_n + score_n)
                scale_end = target - pd.Timedelta(days=score_n + 1)
                score_start = target - pd.Timedelta(days=score_n)
                score_end = target - pd.Timedelta(days=1)
                q_hours = np.full(LOAD_HOURS, np.nan)
            else:
                indexes = _window_indices(date_to_index, target, bonf_n, 1)
                window = np.abs(residual[indexes])
                per_hour_alpha = alpha / LOAD_HOURS
                rank = finite_sample_rank(bonf_n, per_hour_alpha, cap=False)
                if rank > bonf_n:
                    raise AssertionError(
                        f"Bonferroni rank {rank} exceeds calibration count {bonf_n} on {target:%Y-%m-%d}"
                    )
                q_hours = quantile_higher(window, rank).astype(float)
                lower = predicted[row_index] - q_hours
                upper = predicted[row_index] + q_hours
                sigma = np.full(LOAD_HOURS, np.nan)
                q = np.nan
                scale_start = pd.NaT
                scale_end = pd.NaT
                score_start = target - pd.Timedelta(days=bonf_n)
                score_end = target - pd.Timedelta(days=1)
            record: dict[str, Any] = {
                "target_date": target,
                "dataset": data.iloc[row_index]["dataset"],
                "method": method,
                "alpha": alpha,
                "quantile_rank": rank,
                "quantile_scalar": q,
                "scale_floor_mw": floor if method != "bonferroni_365" else np.nan,
                "scale_start": scale_start,
                "scale_end": scale_end,
                "score_start": score_start,
                "score_end": score_end,
                "score_count": primary_n if method in ("standardized_max_90", "one_sided_max_90") else score_n if method == "disjoint_standardized_max_90" else bonf_n,
                "calibration_available_before_target": bool(pd.Timestamp(score_end) < target),
            }
            for hour_index in range(LOAD_HOURS):
                hour = hour_index + 1
                record[f"lower_h{hour:02d}"] = lower[hour_index]
                record[f"upper_h{hour:02d}"] = upper[hour_index]
                record[f"sigma_h{hour:02d}"] = sigma[hour_index]
                record[f"quantile_h{hour:02d}"] = q_hours[hour_index]
            rows.append(record)
    result = pd.DataFrame(rows)
    if not result["calibration_available_before_target"].all():
        raise AssertionError("At least one conformal row uses unavailable residuals")
    return result


def historical_peak_threshold(
    daily: pd.DataFrame,
    cutoff: pd.Timestamp,
    percentile: float,
) -> float:
    available = daily.get("available", pd.Series(True, index=daily.index)).astype(bool)
    dates = pd.to_datetime(daily["date"])
    mask = available & (dates < cutoff)
    peaks = daily.loc[mask, LOAD_COLUMNS].max(axis=1).to_numpy(float)
    if not len(peaks):
        raise AssertionError(f"No historical peaks before {cutoff}")
    return float(np.quantile(peaks, percentile / 100.0, method="linear"))


def band_metrics_for_period(
    store: pd.DataFrame,
    bands: pd.DataFrame,
    daily: pd.DataFrame,
    method: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    peak_threshold_percentile: float = 95.0,
) -> dict[str, Any]:
    point = store.loc[pd.to_datetime(store["target_date"]).between(start, end)].copy()
    band = bands.loc[(bands["method"] == method) & pd.to_datetime(bands["target_date"]).between(start, end)].copy()
    merged = point.merge(band, on=["target_date", "dataset"], how="inner", validate="one_to_one")
    if not len(merged):
        raise AssertionError(f"No band rows for {method}, {start} to {end}")
    actual = merged[TRUE_COLUMNS].to_numpy(float)
    predicted = merged[PRED_COLUMNS].to_numpy(float)
    upper = merged[[f"upper_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]].to_numpy(float)
    threshold = historical_peak_threshold(daily, start, peak_threshold_percentile)
    peaks = actual.max(axis=1)
    peak_events = peaks > threshold
    actual_peak_hours = np.argmax(actual, axis=1)
    upper_covered = actual <= upper
    upper_hourly = float(np.mean(upper_covered))
    upper_daily = float(np.mean(np.all(upper_covered, axis=1)))
    upper_peak_hour = float(np.mean(upper_covered[np.arange(len(actual)), actual_peak_hours]))
    upper_peak_day = float(np.mean(np.all(upper_covered[peak_events], axis=1))) if peak_events.any() else np.nan
    upper_margin = float(np.mean(upper - predicted))
    tau = 0.90
    upper_pinball = float(np.mean((tau - (actual < upper).astype(float)) * (actual - upper)))
    record: dict[str, Any] = {
        "method": method,
        "period": "2026H1" if start == pd.Timestamp("2026-01-01") else str(start.year),
        "start": start,
        "end": end,
        "days": len(merged),
        "peak_threshold_percentile": peak_threshold_percentile,
        "peak_threshold_mw": threshold,
        "peak_event_count": int(peak_events.sum()),
        "hourly_upper_coverage_pct": upper_hourly * 100,
        "daily_upper_coverage_pct": upper_daily * 100,
        "peak_hour_upper_coverage_pct": upper_peak_hour * 100,
        "peak_day_upper_coverage_pct": upper_peak_day * 100 if np.isfinite(upper_peak_day) else np.nan,
        "mean_upper_margin_mw": upper_margin,
        "upper_pinball_loss_mw": upper_pinball,
        **point_metrics(actual, predicted),
    }
    if method == "one_sided_max_90":
        record.update(
            {
                "hourly_coverage_pct": np.nan,
                "daily_simultaneous_coverage_pct": np.nan,
                "peak_hour_coverage_pct": np.nan,
                "peak_day_simultaneous_coverage_pct": np.nan,
                "mean_width_mw": np.nan,
                "interval_score_mw": np.nan,
            }
        )
    else:
        lower = merged[[f"lower_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]].to_numpy(float)
        covered = (actual >= lower) & (actual <= upper)
        below = actual < lower
        above = actual > upper
        alpha = float(merged["alpha"].iloc[0])
        score = (upper - lower) + 2 / alpha * (lower - actual) * below + 2 / alpha * (actual - upper) * above
        record.update(
            {
                "hourly_coverage_pct": float(np.mean(covered) * 100),
                "daily_simultaneous_coverage_pct": float(np.mean(np.all(covered, axis=1)) * 100),
                "peak_hour_coverage_pct": float(np.mean(covered[np.arange(len(actual)), actual_peak_hours]) * 100),
                "peak_day_simultaneous_coverage_pct": float(np.mean(np.all(covered[peak_events], axis=1)) * 100) if peak_events.any() else np.nan,
                "mean_width_mw": float(np.mean(upper - lower)),
                "interval_score_mw": float(np.mean(score)),
            }
        )
    return record


def build_band_metrics(store: pd.DataFrame, bands: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in sorted(bands["method"].unique()):
        for label, start, end in evaluation_periods():
            record = band_metrics_for_period(store, bands, daily, method, start, end)
            record["period"] = label
            rows.append(record)
    return pd.DataFrame(rows)


def rolling_thresholds(daily: pd.DataFrame, target_dates: Iterable[pd.Timestamp], window: int = 365, percentile: float = 95.0) -> np.ndarray:
    available = daily.get("available", pd.Series(True, index=daily.index)).astype(bool)
    profile = daily.loc[available, ["date", *LOAD_COLUMNS]].copy()
    profile["date"] = pd.to_datetime(profile["date"])
    peak_map = dict(zip(profile["date"], profile[LOAD_COLUMNS].max(axis=1).to_numpy(float)))
    values = []
    for target in pd.to_datetime(list(target_dates)):
        history_dates = [target - pd.Timedelta(days=lag) for lag in range(window, 0, -1)]
        missing = [date for date in history_dates if date not in peak_map]
        if missing:
            raise KeyError(f"Rolling threshold for {target:%Y-%m-%d} lacks {len(missing)} dates")
        values.append(float(np.quantile([peak_map[date] for date in history_dates], percentile / 100.0)))
    return np.asarray(values, dtype=float)


def confusion(event: np.ndarray, alert: np.ndarray, cost_ratio: float) -> dict[str, Any]:
    event = event.astype(bool)
    alert = alert.astype(bool)
    tp = int(np.sum(event & alert))
    fp = int(np.sum(~event & alert))
    fn = int(np.sum(event & ~alert))
    tn = int(np.sum(~event & ~alert))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    far = fp / (fp + tn) if fp + tn else np.nan
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "alerts": int(alert.sum()),
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": far,
        "cost": float(fp + cost_ratio * fn),
    }


def tune_parameter(
    event: np.ndarray,
    alert_for_parameter,
    parameters: Iterable[float],
    cost_ratio: float,
) -> tuple[float, dict[str, Any]]:
    candidates = []
    for parameter in parameters:
        metrics = confusion(event, alert_for_parameter(float(parameter)), cost_ratio)
        candidates.append((metrics["cost"], metrics["FN"], metrics["FP"], abs(float(parameter)), float(parameter), metrics))
    candidates.sort(key=lambda item: item[:5])
    best = candidates[0]
    return best[4], best[5]


def exact_top_k(values: np.ndarray, dates: np.ndarray, k: int) -> np.ndarray:
    n = len(values)
    result = np.zeros(n, dtype=bool)
    if k <= 0:
        return result
    if k >= n:
        return np.ones(n, dtype=bool)
    order = np.lexsort((pd.to_datetime(dates).view("int64"), -values))
    result[order[:k]] = True
    return result


def deployable_rank_cutoff(values: np.ndarray, k: int) -> float:
    descending = np.sort(values)[::-1]
    if k <= 0:
        return float("inf")
    if k >= len(descending):
        return float("-inf")
    return float((descending[k - 1] + descending[k]) / 2.0)


def _threshold_arrays(
    daily: pd.DataFrame,
    validation_dates: pd.Series,
    evaluation_dates: pd.Series,
    threshold_spec: str,
    cutoff: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    if threshold_spec == "rolling_365d_95":
        return (
            rolling_thresholds(daily, validation_dates, 365, 95.0),
            rolling_thresholds(daily, evaluation_dates, 365, 95.0),
            None,
        )
    percentile = float(threshold_spec.removeprefix("historical_q"))
    value = historical_peak_threshold(daily, cutoff, percentile)
    return (
        np.full(len(validation_dates), value, dtype=float),
        np.full(len(evaluation_dates), value, dtype=float),
        value,
    )


def decision_analysis(
    store: pd.DataFrame,
    bands: pd.DataFrame,
    daily: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_start = pd.Timestamp(config["validation_start"])
    validation_end = pd.Timestamp(config["validation_end"])
    evaluation_start = pd.Timestamp(config["evaluation_start"])
    evaluation_end = pd.Timestamp(config["evaluation_end"])
    store = store.copy()
    store["target_date"] = pd.to_datetime(store["target_date"])
    point_val = store.loc[store["target_date"].between(validation_start, validation_end)].sort_values("target_date")
    point_test = store.loc[store["target_date"].between(evaluation_start, evaluation_end)].sort_values("target_date")
    if len(point_val) != 366 or len(point_test) != 365:
        raise AssertionError(f"Decision periods require 366 validation and 365 evaluation days; found {len(point_val)}, {len(point_test)}")
    val_dates = point_val["target_date"].reset_index(drop=True)
    test_dates = point_test["target_date"].reset_index(drop=True)
    actual_val = point_val[TRUE_COLUMNS].to_numpy(float)
    actual_test = point_test[TRUE_COLUMNS].to_numpy(float)
    pred_val = point_val[PRED_COLUMNS].to_numpy(float)
    pred_test = point_test[PRED_COLUMNS].to_numpy(float)
    actual_peak_val = actual_val.max(axis=1)
    actual_peak_test = actual_test.max(axis=1)
    predicted_peak_val = pred_val.max(axis=1)
    predicted_peak_test = pred_test.max(axis=1)

    band_upper: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    method_map = {
        "two_sided": "standardized_max_90",
        "one_sided": "one_sided_max_90",
    }
    upper_cols = [f"upper_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]
    for label, method in method_map.items():
        selected = bands.loc[bands["method"] == method].copy()
        selected["target_date"] = pd.to_datetime(selected["target_date"])
        val = selected.loc[selected["target_date"].between(validation_start, validation_end)].sort_values("target_date")
        test = selected.loc[selected["target_date"].between(evaluation_start, evaluation_end)].sort_values("target_date")
        if not np.array_equal(val["target_date"].to_numpy(), val_dates.to_numpy()) or not np.array_equal(test["target_date"].to_numpy(), test_dates.to_numpy()):
            raise AssertionError(f"Band dates do not align for {method}")
        band_upper[label] = (val[upper_cols].to_numpy(float), test[upper_cols].to_numpy(float))

    threshold_specs = [f"historical_q{value:g}" for value in config["peak_percentiles"]]
    if config.get("include_rolling_365d_95", False):
        threshold_specs.append("rolling_365d_95")
    lambda_grid = [float(value) for value in config["lambda_grid"]]
    delta_grid = np.arange(
        float(config["delta_start_mw"]),
        float(config["delta_end_mw"]) + 0.5 * float(config["delta_step_mw"]),
        float(config["delta_step_mw"]),
    )
    cube_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []

    for threshold_spec in threshold_specs:
        threshold_val, threshold_test, fixed_threshold = _threshold_arrays(
            daily, val_dates, test_dates, threshold_spec, validation_start
        )
        event_val = actual_peak_val > threshold_val
        event_test = actual_peak_test > threshold_test
        for cost_ratio_raw in config["cost_ratios"]:
            cost_ratio = float(cost_ratio_raw)
            delta, delta_validation_metrics = tune_parameter(
                event_val,
                lambda value: predicted_peak_val + value > threshold_val,
                delta_grid,
                cost_ratio,
            )
            tuned_point_val = predicted_peak_val + delta > threshold_val
            tuned_point_test = predicted_peak_test + delta > threshold_test
            band_alerts: dict[str, dict[str, np.ndarray | float | dict[str, Any]]] = {}
            for band_label in ("two_sided", "one_sided"):
                upper_val, upper_test = band_upper[band_label]
                margin_val = upper_val - pred_val
                margin_test = upper_test - pred_test
                lambda_value, validation_metrics = tune_parameter(
                    event_val,
                    lambda value, pv=pred_val, mv=margin_val: np.max(pv + value * mv, axis=1) > threshold_val,
                    lambda_grid,
                    cost_ratio,
                )
                tuned_val_score = np.max(pred_val + lambda_value * margin_val, axis=1)
                tuned_test_score = np.max(pred_test + lambda_value * margin_test, axis=1)
                tuned_val_alert = tuned_val_score > threshold_val
                tuned_test_alert = tuned_test_score > threshold_test
                full_val_score = np.max(upper_val, axis=1)
                full_test_score = np.max(upper_test, axis=1)
                full_val_alert = full_val_score > threshold_val
                full_test_alert = full_test_score > threshold_test
                rank_k = int(tuned_val_alert.sum())
                rank_cutoff = deployable_rank_cutoff(predicted_peak_val, rank_k)
                validation_quantile_val = predicted_peak_val >= rank_cutoff
                validation_quantile_test = predicted_peak_test >= rank_cutoff
                equal_count_test = exact_top_k(predicted_peak_test, test_dates.to_numpy(), int(tuned_test_alert.sum()))
                band_alerts[band_label] = {
                    "lambda": lambda_value,
                    "validation_metrics": validation_metrics,
                    "tuned_val": tuned_val_alert,
                    "tuned_test": tuned_test_alert,
                    "tuned_val_score": tuned_val_score,
                    "tuned_test_score": tuned_test_score,
                    "full_val": full_val_alert,
                    "full_test": full_test_alert,
                    "full_val_score": full_val_score,
                    "full_test_score": full_test_score,
                    "rank_cutoff": rank_cutoff,
                    "validation_quantile_val": validation_quantile_val,
                    "validation_quantile_test": validation_quantile_test,
                    "equal_count_test": equal_count_test,
                }

            methods: dict[str, tuple[np.ndarray, np.ndarray, str, float | str, bool, str | None]] = {
                "fixed_point": (
                    predicted_peak_val > threshold_val,
                    predicted_peak_test > threshold_test,
                    "none",
                    0.0,
                    False,
                    None,
                ),
                "tuned_point": (tuned_point_val, tuned_point_test, "delta_mw", delta, False, None),
            }
            for band_label in ("two_sided", "one_sided"):
                details = band_alerts[band_label]
                methods[f"{band_label}_conformal"] = (
                    details["tuned_val"],
                    details["tuned_test"],
                    "lambda",
                    float(details["lambda"]),
                    False,
                    band_label,
                )
                methods[f"full_{band_label}_conformal"] = (
                    details["full_val"],
                    details["full_test"],
                    "lambda",
                    1.0,
                    False,
                    band_label,
                )
                methods[f"validation_quantile_point_{band_label}"] = (
                    details["validation_quantile_val"],
                    details["validation_quantile_test"],
                    "predicted_peak_cutoff_mw",
                    float(details["rank_cutoff"]),
                    False,
                    band_label,
                )
                methods[f"equal_alert_point_{band_label}"] = (
                    np.zeros(len(event_val), dtype=bool),
                    details["equal_count_test"],
                    "test_alert_count",
                    int(np.asarray(details["equal_count_test"]).sum()),
                    True,
                    band_label,
                )

            for method, (alerts_val, alerts_test, setting_type, setting, ex_post, source_band) in methods.items():
                test_metrics = confusion(event_test, np.asarray(alerts_test), cost_ratio)
                val_metrics = confusion(event_val, np.asarray(alerts_val), cost_ratio) if method.startswith("equal_alert") is False else {key: np.nan for key in ("TP", "FP", "FN", "TN", "alerts", "precision", "recall", "false_alarm_rate", "cost")}
                cube_rows.append(
                    {
                        "method": method,
                        "source_band": source_band,
                        "peak_threshold": threshold_spec,
                        "fixed_threshold_mw": fixed_threshold,
                        "cost_ratio": cost_ratio,
                        "setting_type": setting_type,
                        "setting": setting,
                        "is_ex_post": ex_post,
                        **test_metrics,
                    }
                )
                tuning_rows.append(
                    {
                        "method": method,
                        "source_band": source_band,
                        "peak_threshold": threshold_spec,
                        "cost_ratio": cost_ratio,
                        "setting_type": setting_type,
                        "selected_setting": setting,
                        "validation_TP": val_metrics["TP"],
                        "validation_FP": val_metrics["FP"],
                        "validation_FN": val_metrics["FN"],
                        "validation_TN": val_metrics["TN"],
                        "validation_alerts": val_metrics["alerts"],
                        "validation_cost": val_metrics["cost"],
                        "is_ex_post": ex_post,
                    }
                )
                if method == "fixed_point" or method == "tuned_point" or method.startswith("validation_quantile") or method.startswith("equal_alert"):
                    score_test = predicted_peak_test + (delta if method == "tuned_point" else 0.0)
                elif method.startswith("full_two"):
                    score_test = np.asarray(band_alerts["two_sided"]["full_test_score"])
                elif method.startswith("full_one"):
                    score_test = np.asarray(band_alerts["one_sided"]["full_test_score"])
                elif method.startswith("two_sided"):
                    score_test = np.asarray(band_alerts["two_sided"]["tuned_test_score"])
                else:
                    score_test = np.asarray(band_alerts["one_sided"]["tuned_test_score"])
                alerts_test_array = np.asarray(alerts_test).astype(bool)
                for index, date in enumerate(test_dates):
                    false_alarm = bool(alerts_test_array[index] and not event_test[index])
                    missed = bool(event_test[index] and not alerts_test_array[index])
                    daily_rows.append(
                        {
                            "target_date": date,
                            "method": method,
                            "source_band": source_band,
                            "peak_threshold": threshold_spec,
                            "cost_ratio": cost_ratio,
                            "setting_type": setting_type,
                            "setting": setting,
                            "is_ex_post": ex_post,
                            "actual_peak_mw": actual_peak_test[index],
                            "predicted_peak_mw": predicted_peak_test[index],
                            "decision_score_mw": score_test[index],
                            "threshold_mw": threshold_test[index],
                            "event": bool(event_test[index]),
                            "alert": bool(alerts_test_array[index]),
                            "false_alarm": false_alarm,
                            "missed_event": missed,
                            "daily_cost": float(false_alarm + cost_ratio * missed),
                        }
                    )
    return pd.DataFrame(cube_rows), pd.DataFrame(daily_rows), pd.DataFrame(tuning_rows)


def circular_block_weights(n: int, replicates: int, block_length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(n / block_length))
    starts = rng.integers(0, n, size=(replicates, blocks), dtype=np.int32)
    offsets = np.arange(block_length, dtype=np.int32)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    indices = indices.reshape(replicates, -1)[:, :n]
    weights = np.zeros((replicates, n), dtype=np.int16)
    rows = np.repeat(np.arange(replicates, dtype=np.int32), n)
    np.add.at(weights, (rows, indices.ravel()), 1)
    return weights


def paired_block_bootstrap(
    daily_decisions: pd.DataFrame,
    config: dict[str, Any],
    random_seed: int,
) -> pd.DataFrame:
    comparisons = [
        ("two_sided_conformal", "tuned_point"),
        ("one_sided_conformal", "tuned_point"),
        ("two_sided_conformal", "equal_alert_point_two_sided"),
        ("one_sided_conformal", "equal_alert_point_one_sided"),
        ("two_sided_conformal", "validation_quantile_point_two_sided"),
        ("one_sided_conformal", "validation_quantile_point_one_sided"),
    ]
    replicates = int(config["replicates"])
    confidence = float(config["confidence_level"])
    alpha_tail = (1 - confidence) / 2
    rows: list[dict[str, Any]] = []
    dates = np.sort(pd.to_datetime(daily_decisions["target_date"].unique()))
    n = len(dates)
    if n != 365:
        raise AssertionError(f"Expected 365 decision days; found {n}")
    weights_by_block = {
        int(block): circular_block_weights(n, replicates, int(block), random_seed + int(block))
        for block in config["block_lengths_days"]
    }
    for (threshold, cost_ratio), group in daily_decisions.groupby(["peak_threshold", "cost_ratio"], sort=True):
        group = group.copy()
        group["target_date"] = pd.to_datetime(group["target_date"])
        by_method = {method: frame.sort_values("target_date") for method, frame in group.groupby("method")}
        for method_a, method_b in comparisons:
            if method_a not in by_method or method_b not in by_method:
                continue
            a = by_method[method_a]
            b = by_method[method_b]
            if not np.array_equal(a["target_date"].to_numpy(), dates) or not np.array_equal(b["target_date"].to_numpy(), dates):
                raise AssertionError(f"Decision dates do not align for {method_a} vs {method_b}")
            event = a["event"].to_numpy(bool)
            alert_a = a["alert"].to_numpy(bool)
            alert_b = b["alert"].to_numpy(bool)
            cost_a = a["daily_cost"].to_numpy(float)
            cost_b = b["daily_cost"].to_numpy(float)
            vectors = {
                "tp_a": (event & alert_a).astype(np.int16),
                "fn_a": (event & ~alert_a).astype(np.int16),
                "fp_a": (~event & alert_a).astype(np.int16),
                "tn_a": (~event & ~alert_a).astype(np.int16),
                "tp_b": (event & alert_b).astype(np.int16),
                "fn_b": (event & ~alert_b).astype(np.int16),
                "fp_b": (~event & alert_b).astype(np.int16),
                "tn_b": (~event & ~alert_b).astype(np.int16),
            }
            observed_a = confusion(event, alert_a, float(cost_ratio))
            observed_b = confusion(event, alert_b, float(cost_ratio))
            observed = {
                "cost": observed_a["cost"] - observed_b["cost"],
                "recall": observed_a["recall"] - observed_b["recall"],
                "false_alarm_rate": observed_a["false_alarm_rate"] - observed_b["false_alarm_rate"],
            }
            for block_length, weights in weights_by_block.items():
                cost_diff = weights @ (cost_a - cost_b)
                tp_a = weights @ vectors["tp_a"]
                fn_a = weights @ vectors["fn_a"]
                fp_a = weights @ vectors["fp_a"]
                tn_a = weights @ vectors["tn_a"]
                tp_b = weights @ vectors["tp_b"]
                fn_b = weights @ vectors["fn_b"]
                fp_b = weights @ vectors["fp_b"]
                tn_b = weights @ vectors["tn_b"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    recall_diff = tp_a / (tp_a + fn_a) - tp_b / (tp_b + fn_b)
                    far_diff = fp_a / (fp_a + tn_a) - fp_b / (fp_b + tn_b)
                bootstrap_values = {
                    "cost": cost_diff.astype(float),
                    "recall": recall_diff.astype(float),
                    "false_alarm_rate": far_diff.astype(float),
                }
                for metric, values in bootstrap_values.items():
                    finite = values[np.isfinite(values)]
                    rows.append(
                        {
                            "peak_threshold": threshold,
                            "cost_ratio": float(cost_ratio),
                            "method_a": method_a,
                            "method_b": method_b,
                            "difference_definition": "method_a_minus_method_b",
                            "metric": metric,
                            "block_length_days": block_length,
                            "replicates_requested": replicates,
                            "replicates_valid": len(finite),
                            "observed_difference": observed[metric],
                            "bootstrap_mean": float(np.mean(finite)),
                            "ci_lower": float(np.quantile(finite, alpha_tail)),
                            "ci_upper": float(np.quantile(finite, 1 - alpha_tail)),
                            "probability_difference_lt_zero": float(np.mean(finite < 0)),
                            "probability_difference_gt_zero": float(np.mean(finite > 0)),
                        }
                    )
    return pd.DataFrame(rows)


def paired_point_model_bootstrap(
    reference: pd.DataFrame,
    comparators: dict[str, pd.DataFrame],
    config: dict[str, Any],
    random_seed: int,
    evaluation_start: str | pd.Timestamp = "2025-01-01",
    evaluation_end: str | pd.Timestamp = "2025-12-31",
) -> pd.DataFrame:
    """Paired moving-block uncertainty for point-model errors on identical dates.

    Differences are comparator minus reference, so positive MAE/RMSE/peak-MAE
    values favor the reference model.
    """
    start = pd.Timestamp(evaluation_start)
    end = pd.Timestamp(evaluation_end)
    reference = reference.copy()
    reference["target_date"] = pd.to_datetime(reference["target_date"])
    reference = reference.loc[reference["target_date"].between(start, end)].sort_values("target_date")
    dates = reference["target_date"].to_numpy()
    n = len(reference)
    if n != 365:
        raise AssertionError(f"Point-model bootstrap expected 365 reference days; found {n}")
    reference_true = reference[TRUE_COLUMNS].to_numpy(float)
    reference_pred = reference[PRED_COLUMNS].to_numpy(float)
    ref_error = reference_true - reference_pred
    ref_daily_mae = np.mean(np.abs(ref_error), axis=1)
    ref_daily_mse = np.mean(ref_error**2, axis=1)
    ref_daily_peak = np.abs(reference_true.max(axis=1) - reference_pred.max(axis=1))
    replicates = int(config["replicates"])
    confidence = float(config["confidence_level"])
    tail = (1 - confidence) / 2
    weights_by_block = {
        int(block): circular_block_weights(n, replicates, int(block), random_seed + 10_000 + int(block))
        for block in config["block_lengths_days"]
    }
    rows: list[dict[str, Any]] = []

    for model, frame in comparators.items():
        candidate = frame.copy()
        candidate["target_date"] = pd.to_datetime(candidate["target_date"])
        candidate = candidate.loc[candidate["target_date"].between(start, end)].sort_values("target_date")
        if not np.array_equal(candidate["target_date"].to_numpy(), dates):
            raise AssertionError(f"Point-model bootstrap dates do not align for {model}")
        true = candidate[TRUE_COLUMNS].to_numpy(float)
        if not np.allclose(true, reference_true, rtol=0, atol=1e-6):
            raise AssertionError(f"Point-model bootstrap targets do not align for {model}")
        predicted = candidate[PRED_COLUMNS].to_numpy(float)
        error = true - predicted
        daily_mae = np.mean(np.abs(error), axis=1)
        daily_mse = np.mean(error**2, axis=1)
        daily_peak = np.abs(true.max(axis=1) - predicted.max(axis=1))
        observed = {
            "mae_mw": float(np.mean(daily_mae) - np.mean(ref_daily_mae)),
            "rmse_mw": float(np.sqrt(np.mean(daily_mse)) - np.sqrt(np.mean(ref_daily_mse))),
            "peak_mae_mw": float(np.mean(daily_peak) - np.mean(ref_daily_peak)),
        }
        for block_length, weights in weights_by_block.items():
            denominator = weights.sum(axis=1).astype(float)
            bootstrap_values = {
                "mae_mw": (weights @ daily_mae) / denominator - (weights @ ref_daily_mae) / denominator,
                "rmse_mw": np.sqrt((weights @ daily_mse) / denominator)
                - np.sqrt((weights @ ref_daily_mse) / denominator),
                "peak_mae_mw": (weights @ daily_peak) / denominator - (weights @ ref_daily_peak) / denominator,
            }
            for metric, values in bootstrap_values.items():
                rows.append(
                    {
                        "model_a": model,
                        "model_b": "Ridge",
                        "difference_definition": "model_a_minus_model_b",
                        "metric": metric,
                        "block_length_days": int(block_length),
                        "replicates": replicates,
                        "observed_difference": observed[metric],
                        "bootstrap_mean": float(np.mean(values)),
                        "ci_lower": float(np.quantile(values, tail)),
                        "ci_upper": float(np.quantile(values, 1 - tail)),
                        "probability_difference_lt_zero": float(np.mean(values < 0)),
                        "probability_difference_gt_zero": float(np.mean(values > 0)),
                    }
                )
    return pd.DataFrame(rows)


def seasonal_naive_results(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = daily.get("available", pd.Series(True, index=daily.index)).astype(bool)
    data = daily.loc[available, ["date", *LOAD_COLUMNS]].copy()
    data["date"] = pd.to_datetime(data["date"])
    profile_map = {pd.Timestamp(row.date): np.asarray([getattr(row, col) for col in LOAD_COLUMNS], dtype=float) for row in data.itertuples(index=False)}
    prediction_rows = []
    metric_rows = []
    for lag in (1, 7):
        rows = []
        for date in data["date"]:
            if date - pd.Timedelta(days=lag) not in profile_map:
                continue
            actual = profile_map[date]
            predicted = profile_map[date - pd.Timedelta(days=lag)]
            record: dict[str, Any] = {"target_date": date, "model": f"D-{lag}"}
            for index in range(LOAD_HOURS):
                record[TRUE_COLUMNS[index]] = actual[index]
                record[PRED_COLUMNS[index]] = predicted[index]
            rows.append(record)
        frame = pd.DataFrame(rows)
        prediction_rows.append(frame)
        for label, start, end in evaluation_periods():
            selected = frame.loc[pd.to_datetime(frame["target_date"]).between(start, end)]
            values = point_metrics(selected[TRUE_COLUMNS].to_numpy(float), selected[PRED_COLUMNS].to_numpy(float))
            metric_rows.append({"model": f"D-{lag}", "period": label, "days": len(selected), **values})
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(metric_rows)


def lambda_stability(
    store: pd.DataFrame,
    bands: pd.DataFrame,
    daily: pd.DataFrame,
    cost_ratios: Iterable[float],
    lambda_grid: Iterable[float],
) -> pd.DataFrame:
    store = store.copy()
    store["target_date"] = pd.to_datetime(store["target_date"])
    upper_cols = [f"upper_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]
    rows = []
    for year in range(2020, 2025):
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        point = store.loc[store["target_date"].between(start, end)].sort_values("target_date")
        actual = point[TRUE_COLUMNS].to_numpy(float)
        predicted = point[PRED_COLUMNS].to_numpy(float)
        threshold = historical_peak_threshold(daily, start, 95.0)
        event = actual.max(axis=1) > threshold
        for label, method in (("two_sided", "standardized_max_90"), ("one_sided", "one_sided_max_90")):
            band = bands.loc[(bands["method"] == method) & pd.to_datetime(bands["target_date"]).between(start, end)].sort_values("target_date")
            if not np.array_equal(point["target_date"].to_numpy(), pd.to_datetime(band["target_date"]).to_numpy()):
                raise AssertionError(f"Lambda-stability dates do not align for {year} {method}")
            margin = band[upper_cols].to_numpy(float) - predicted
            for ratio in cost_ratios:
                selected, met = tune_parameter(
                    event,
                    lambda value, p=predicted, m=margin: np.max(p + value * m, axis=1) > threshold,
                    lambda_grid,
                    float(ratio),
                )
                rows.append(
                    {
                        "year": year,
                        "band": label,
                        "cost_ratio": float(ratio),
                        "selected_lambda": selected,
                        "peak_threshold_mw": threshold,
                        "peak_events": int(event.sum()),
                        **met,
                    }
                )
    return pd.DataFrame(rows)


def multiyear_point_metrics(store: pd.DataFrame, model_name: str = "Ridge") -> pd.DataFrame:
    store = store.copy()
    store["target_date"] = pd.to_datetime(store["target_date"])
    rows = []
    for label, start, end in evaluation_periods():
        selected = store.loc[store["target_date"].between(start, end)]
        rows.append(
            {
                "model": model_name,
                "period": label,
                "days": len(selected),
                **point_metrics(selected[TRUE_COLUMNS].to_numpy(float), selected[PRED_COLUMNS].to_numpy(float)),
            }
        )
    return pd.DataFrame(rows)


def output_hash_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"results_hashes.csv", "hash_manifest_status.json"}:
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return pd.DataFrame(rows)
