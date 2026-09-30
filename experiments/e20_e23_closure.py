from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "work" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

from experiments.revision_common import (
    LOAD_HOURS,
    PRED_COLUMNS,
    RESIDUAL_COLUMNS,
    TARGET_COLUMNS,
    TRUE_COLUMNS,
    band_metrics_for_period,
    circular_block_weights,
    finite_sample_rank,
    generate_oos_ridge,
    historical_peak_threshold,
    point_metrics,
    quantile_higher,
    write_csv,
    write_json,
    write_parquet,
)


LOWER_COLUMNS = [f"lower_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]
UPPER_COLUMNS = [f"upper_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]
SIGMA_COLUMNS = [f"sigma_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]
QUANTILE_COLUMNS = [f"quantile_h{hour:02d}" for hour in range(1, LOAD_HOURS + 1)]


def _window_indices(
    date_to_index: dict[pd.Timestamp, int], target: pd.Timestamp, history_days: int
) -> list[int] | None:
    dates = [target - pd.Timedelta(days=lag) for lag in range(history_days, 0, -1)]
    if any(date not in date_to_index for date in dates):
        return None
    return [date_to_index[date] for date in dates]


def _band_record(
    target: pd.Timestamp,
    dataset: str,
    method: str,
    alpha: float,
    rank: int,
    lower: np.ndarray,
    upper: np.ndarray,
    sigma: np.ndarray,
    quantiles: np.ndarray,
    scale_start: pd.Timestamp | pd.NaT,
    scale_end: pd.Timestamp | pd.NaT,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
    score_count: int,
    quantile_scalar: float = np.nan,
    scale_floor_mw: float = np.nan,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "target_date": target,
        "dataset": dataset,
        "method": method,
        "alpha": alpha,
        "quantile_rank": rank,
        "quantile_scalar": quantile_scalar,
        "scale_floor_mw": scale_floor_mw,
        "scale_start": scale_start,
        "scale_end": scale_end,
        "score_start": score_start,
        "score_end": score_end,
        "score_count": score_count,
        "calibration_available_before_target": bool(pd.Timestamp(score_end) < target),
    }
    for hour_index in range(LOAD_HOURS):
        record[LOWER_COLUMNS[hour_index]] = lower[hour_index]
        record[UPPER_COLUMNS[hour_index]] = upper[hour_index]
        record[SIGMA_COLUMNS[hour_index]] = sigma[hour_index]
        record[QUANTILE_COLUMNS[hour_index]] = quantiles[hour_index]
    return record


def build_rolling_standardized_bands(
    store: pd.DataFrame,
    window_days: int,
    floor_mw: float,
    alpha: float,
    method: str,
) -> pd.DataFrame:
    data = store.sort_values("target_date").reset_index(drop=True).copy()
    data["target_date"] = pd.to_datetime(data["target_date"])
    residual = data[RESIDUAL_COLUMNS].to_numpy(float)
    predicted = data[PRED_COLUMNS].to_numpy(float)
    date_to_index = {pd.Timestamp(value): index for index, value in enumerate(data["target_date"])}
    rank = finite_sample_rank(window_days, alpha, cap=True)
    rows: list[dict[str, Any]] = []
    for row_index, target in enumerate(data["target_date"]):
        indexes = _window_indices(date_to_index, target, window_days)
        if indexes is None:
            continue
        window = residual[indexes]
        sigma = np.maximum(float(floor_mw), np.median(np.abs(window), axis=0))
        scores = np.max(np.abs(window) / sigma, axis=1)
        q = float(quantile_higher(scores[:, None], rank)[0])
        margin = q * sigma
        rows.append(
            _band_record(
                target,
                str(data.iloc[row_index]["dataset"]),
                method,
                alpha,
                rank,
                predicted[row_index] - margin,
                predicted[row_index] + margin,
                sigma,
                np.full(LOAD_HOURS, np.nan),
                target - pd.Timedelta(days=window_days),
                target - pd.Timedelta(days=1),
                target - pd.Timedelta(days=window_days),
                target - pd.Timedelta(days=1),
                window_days,
                q,
                floor_mw,
            )
        )
    return pd.DataFrame(rows)


def build_rolling_raw_max_band(
    store: pd.DataFrame, window_days: int = 90, alpha: float = 0.10
) -> pd.DataFrame:
    data = store.sort_values("target_date").reset_index(drop=True).copy()
    data["target_date"] = pd.to_datetime(data["target_date"])
    residual = data[RESIDUAL_COLUMNS].to_numpy(float)
    predicted = data[PRED_COLUMNS].to_numpy(float)
    date_to_index = {pd.Timestamp(value): index for index, value in enumerate(data["target_date"])}
    rank = finite_sample_rank(window_days, alpha, cap=True)
    rows: list[dict[str, Any]] = []
    for row_index, target in enumerate(data["target_date"]):
        indexes = _window_indices(date_to_index, target, window_days)
        if indexes is None:
            continue
        scores = np.max(np.abs(residual[indexes]), axis=1)
        q = float(quantile_higher(scores[:, None], rank)[0])
        margin = np.full(LOAD_HOURS, q)
        rows.append(
            _band_record(
                target,
                str(data.iloc[row_index]["dataset"]),
                f"rolling_raw_max_{window_days}",
                alpha,
                rank,
                predicted[row_index] - margin,
                predicted[row_index] + margin,
                np.full(LOAD_HOURS, np.nan),
                np.full(LOAD_HOURS, q),
                pd.NaT,
                pd.NaT,
                target - pd.Timedelta(days=window_days),
                target - pd.Timedelta(days=1),
                window_days,
                q,
            )
        )
    return pd.DataFrame(rows)


def build_rolling_marginal_band(
    store: pd.DataFrame, window_days: int = 90, alpha: float = 0.10
) -> pd.DataFrame:
    data = store.sort_values("target_date").reset_index(drop=True).copy()
    data["target_date"] = pd.to_datetime(data["target_date"])
    residual = data[RESIDUAL_COLUMNS].to_numpy(float)
    predicted = data[PRED_COLUMNS].to_numpy(float)
    date_to_index = {pd.Timestamp(value): index for index, value in enumerate(data["target_date"])}
    rank = finite_sample_rank(window_days, alpha, cap=True)
    rows: list[dict[str, Any]] = []
    for row_index, target in enumerate(data["target_date"]):
        indexes = _window_indices(date_to_index, target, window_days)
        if indexes is None:
            continue
        q_hours = quantile_higher(np.abs(residual[indexes]), rank).astype(float)
        rows.append(
            _band_record(
                target,
                str(data.iloc[row_index]["dataset"]),
                f"rolling_marginal_{window_days}",
                alpha,
                rank,
                predicted[row_index] - q_hours,
                predicted[row_index] + q_hours,
                np.full(LOAD_HOURS, np.nan),
                q_hours,
                pd.NaT,
                pd.NaT,
                target - pd.Timedelta(days=window_days),
                target - pd.Timedelta(days=1),
                window_days,
            )
        )
    return pd.DataFrame(rows)


def build_static_previous_year_band(
    store: pd.DataFrame, floor_mw: float = 50.0, alpha: float = 0.10
) -> pd.DataFrame:
    data = store.sort_values("target_date").reset_index(drop=True).copy()
    data["target_date"] = pd.to_datetime(data["target_date"])
    residual = data[RESIDUAL_COLUMNS].to_numpy(float)
    predicted = data[PRED_COLUMNS].to_numpy(float)
    rows: list[dict[str, Any]] = []
    for target_year in sorted(data["target_date"].dt.year.unique()):
        calibration_start = pd.Timestamp(f"{target_year - 1}-01-01")
        calibration_end = pd.Timestamp(f"{target_year - 1}-12-31")
        calibration_mask = data["target_date"].between(calibration_start, calibration_end).to_numpy()
        target_mask = data["target_date"].dt.year.eq(target_year).to_numpy()
        if not calibration_mask.any() or not target_mask.any():
            continue
        window = residual[calibration_mask]
        n = len(window)
        sigma = np.maximum(float(floor_mw), np.median(np.abs(window), axis=0))
        scores = np.max(np.abs(window) / sigma, axis=1)
        rank = finite_sample_rank(n, alpha, cap=True)
        q = float(quantile_higher(scores[:, None], rank)[0])
        margin = q * sigma
        for row_index in np.flatnonzero(target_mask):
            target = pd.Timestamp(data.iloc[row_index]["target_date"])
            rows.append(
                _band_record(
                    target,
                    str(data.iloc[row_index]["dataset"]),
                    "static_standardized_previous_year",
                    alpha,
                    rank,
                    predicted[row_index] - margin,
                    predicted[row_index] + margin,
                    sigma,
                    np.full(LOAD_HOURS, np.nan),
                    calibration_start,
                    calibration_end,
                    calibration_start,
                    calibration_end,
                    n,
                    q,
                    floor_mw,
                )
            )
    result = pd.DataFrame(rows)
    if len(result) and not result["calibration_available_before_target"].all():
        raise AssertionError("Static band uses calibration residuals not available before target")
    return result


def band_metrics_for_periods(
    store: pd.DataFrame,
    bands: pd.DataFrame,
    daily: pd.DataFrame,
    periods: Iterable[tuple[str, pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    rows = []
    for method in sorted(bands["method"].unique()):
        for label, start, end in periods:
            try:
                row = band_metrics_for_period(store, bands, daily, method, start, end)
            except AssertionError:
                continue
            row["period"] = label
            row["peak_threshold_definition"] = "year_specific_historical_q95_before_period_start"
            rows.append(row)
    return pd.DataFrame(rows)


def coverage_bootstrap(
    store: pd.DataFrame,
    bands: pd.DataFrame,
    dataset: str,
    periods: Iterable[tuple[str, pd.Timestamp, pd.Timestamp]],
    method_metrics: dict[str, list[str]],
    config: dict[str, Any],
    random_seed: int,
) -> pd.DataFrame:
    point = store.copy()
    point["target_date"] = pd.to_datetime(point["target_date"])
    bands = bands.copy()
    bands["target_date"] = pd.to_datetime(bands["target_date"])
    replicates = int(config["replicates"])
    nominal = float(config["nominal_coverage"])
    tail = (1 - float(config["confidence_level"])) / 2
    rows: list[dict[str, Any]] = []
    for method, metrics in method_metrics.items():
        method_band = bands.loc[bands["method"] == method]
        for period_index, (label, start, end) in enumerate(periods):
            selected_point = point.loc[point["target_date"].between(start, end)]
            selected_band = method_band.loc[method_band["target_date"].between(start, end)]
            merged = selected_point.merge(selected_band, on=["target_date", "dataset"], validate="one_to_one")
            if not len(merged):
                continue
            actual = merged[TRUE_COLUMNS].to_numpy(float)
            upper = merged[UPPER_COLUMNS].to_numpy(float)
            lower = merged[LOWER_COLUMNS].to_numpy(float)
            indicators = {"daily_upper_coverage": np.all(actual <= upper, axis=1).astype(float)}
            if np.isfinite(lower).all():
                indicators["daily_simultaneous_coverage"] = np.all((actual >= lower) & (actual <= upper), axis=1).astype(float)
            n = len(merged)
            for block in config["block_lengths_days"]:
                weights = circular_block_weights(
                    n, replicates, int(block), random_seed + period_index * 1000 + int(block)
                )
                denominator = weights.sum(axis=1).astype(float)
                for metric in metrics:
                    if metric not in indicators:
                        raise AssertionError(f"Metric {metric} unavailable for {method}")
                    values = indicators[metric]
                    boot = (weights @ values) / denominator
                    observed = float(np.mean(values))
                    rows.append(
                        {
                            "dataset": dataset,
                            "period": label,
                            "method": method,
                            "metric": metric,
                            "days": n,
                            "observed_coverage_pct": observed * 100,
                            "nominal_coverage_pct": nominal * 100,
                            "block_length_days": int(block),
                            "replicates": replicates,
                            "ci_lower_pct": float(np.quantile(boot, tail) * 100),
                            "ci_upper_pct": float(np.quantile(boot, 1 - tail) * 100),
                            "bootstrap_mean_pct": float(np.mean(boot) * 100),
                            "bootstrap_fraction_below_nominal": float(np.mean(boot < nominal)),
                        }
                    )
    return pd.DataFrame(rows)


def ridge_development_sensitivity(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    cpu_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(config["start"])
    end = pd.Timestamp(config["end"])
    stores = []
    audits = []
    metrics = []
    for training_years in config["training_windows_years"]:
        for alpha in config["ridge_alphas"]:
            configuration = f"ridge_a{float(alpha):g}_w{int(training_years)}y"
            store, audit = generate_oos_ridge(
                features,
                manifest,
                start,
                end,
                float(alpha),
                int(training_years),
                "ISO_NE",
                cpu_threads,
            )
            compact = store[["target_date", *TRUE_COLUMNS, *PRED_COLUMNS]].copy()
            compact.insert(1, "configuration", configuration)
            compact.insert(2, "ridge_alpha", float(alpha))
            compact.insert(3, "training_window_years", int(training_years))
            stores.append(compact)
            audit.insert(1, "configuration", configuration)
            audits.append(audit)
            store_dates = pd.to_datetime(store["target_date"])
            periods = [("2020-2023", start, end)] + [
                (str(year), pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
                for year in range(2020, 2024)
            ]
            for label, period_start, period_end in periods:
                selected = store.loc[store_dates.between(period_start, period_end)]
                metrics.append(
                    {
                        "configuration": configuration,
                        "ridge_alpha": float(alpha),
                        "training_window_years": int(training_years),
                        "period": label,
                        "days": len(selected),
                        "frozen_primary": bool(
                            float(alpha) == float(config["frozen_primary_alpha"])
                            and int(training_years) == int(config["frozen_primary_training_years"])
                        ),
                        **point_metrics(
                            selected[TRUE_COLUMNS].to_numpy(float),
                            selected[PRED_COLUMNS].to_numpy(float),
                        ),
                    }
                )
    return pd.concat(stores, ignore_index=True), pd.concat(audits, ignore_index=True), pd.DataFrame(metrics)


def development_calibration_sensitivity(
    frozen_store: pd.DataFrame,
    daily: pd.DataFrame,
    config: dict[str, Any],
    alpha: float = 0.10,
    floor_mw: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(config["start"])
    end = pd.Timestamp(config["end"])
    periods = [("2020-2023", start, end)] + [
        (str(year), pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
        for year in range(2020, 2024)
    ]
    bands = []
    metrics = []
    for window in config["calibration_windows_days"]:
        method = f"development_standardized_max_w{int(window)}"
        frame = build_rolling_standardized_bands(
            frozen_store, int(window), float(floor_mw), float(alpha), method
        )
        bands.append(frame)
        for label, period_start, period_end in periods:
            row = band_metrics_for_period(
                frozen_store, frame, daily, method, period_start, period_end
            )
            row["period"] = label
            row["calibration_window_days"] = int(window)
            row["frozen_primary"] = int(window) == int(config["frozen_primary_calibration_days"])
            row["nominal_daily_coverage_pct"] = (1 - alpha) * 100
            row["absolute_daily_coverage_error_pp"] = abs(
                row["daily_simultaneous_coverage_pct"] - (1 - alpha) * 100
            )
            metrics.append(row)
    return pd.concat(bands, ignore_index=True), pd.DataFrame(metrics)


def calibration_window_selection_audit(
    window_metrics: pd.DataFrame,
    validation_period: str = "2024",
    nominal_coverage_pct: float = 90.0,
    originally_frozen_window_days: int = 90,
) -> pd.DataFrame:
    """Create an explicit, non-post-hoc audit of the archived 2024 window results."""
    selected = window_metrics.loc[window_metrics["period"].astype(str) == str(validation_period)].copy()
    selected = selected.sort_values("calibration_window_days").reset_index(drop=True)
    selected["nominal_daily_coverage_pct"] = float(nominal_coverage_pct)
    selected["absolute_daily_coverage_gap_pp"] = (
        selected["daily_simultaneous_coverage_pct"] - float(nominal_coverage_pct)
    ).abs()
    selected["rank_coverage_closeness"] = selected["absolute_daily_coverage_gap_pp"].rank(
        method="min", ascending=True
    ).astype(int)
    selected["rank_mean_width"] = selected["mean_width_mw"].rank(method="min", ascending=True).astype(int)
    selected["rank_interval_score"] = selected["interval_score_mw"].rank(
        method="min", ascending=True
    ).astype(int)
    loss_columns = ["absolute_daily_coverage_gap_pp", "mean_width_mw", "interval_score_mw"]
    dominated = []
    dominators = []
    values = selected[loss_columns].to_numpy(float)
    windows = selected["calibration_window_days"].to_numpy(int)
    for row_index in range(len(selected)):
        other_dominates = np.all(values <= values[row_index] + 1e-12, axis=1) & np.any(
            values < values[row_index] - 1e-12, axis=1
        )
        other_dominates[row_index] = False
        matching = windows[other_dominates].tolist()
        dominated.append(bool(matching))
        dominators.append(",".join(str(value) for value in matching))
    selected["pareto_dominated_on_three_reported_metrics"] = dominated
    selected["dominating_window_days"] = dominators
    selected["originally_frozen_window"] = selected["calibration_window_days"].eq(
        int(originally_frozen_window_days)
    )
    selected["archived_objective_selection_criterion"] = (
        "not recoverable from archived experimental records"
    )
    selected["audit_interpretation"] = np.where(
        selected["originally_frozen_window"],
        "retained original frozen setting; not numerically optimal on the three reported 2024 metrics",
        "validation sensitivity candidate; not selected post hoc",
    )
    return selected


def peak_event_audit(
    store: pd.DataFrame,
    decision_daily: pd.DataFrame,
    decision_cube: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(config["evaluation_start"])
    end = pd.Timestamp(config["evaluation_end"])
    threshold_spec = str(config["peak_threshold"])
    cost_ratio = float(config["cost_ratio"])
    methods = [
        "fixed_point",
        "tuned_point",
        "validation_quantile_point_two_sided",
        "validation_quantile_point_one_sided",
        "two_sided_conformal",
        "one_sided_conformal",
        "equal_alert_point_two_sided",
        "equal_alert_point_one_sided",
    ]
    daily = decision_daily.copy()
    daily["target_date"] = pd.to_datetime(daily["target_date"])
    selected = daily.loc[
        daily["target_date"].between(start, end)
        & daily["peak_threshold"].eq(threshold_spec)
        & daily["cost_ratio"].eq(cost_ratio)
        & daily["method"].isin(methods)
    ].copy()
    event_dates = np.sort(selected.loc[selected["event"], "target_date"].unique())
    if len(event_dates) != 17:
        raise AssertionError(f"Expected 17 ISO-NE peak events; found {len(event_dates)}")
    point = store.copy()
    point["target_date"] = pd.to_datetime(point["target_date"])
    point = point.loc[point["target_date"].isin(event_dates)].sort_values("target_date")
    actual = point[TRUE_COLUMNS].to_numpy(float)
    predicted = point[PRED_COLUMNS].to_numpy(float)
    base = pd.DataFrame(
        {
            "target_date": point["target_date"].to_numpy(),
            "actual_peak_mw": actual.max(axis=1),
            "actual_peak_hour_position": np.argmax(actual, axis=1) + 1,
            "point_peak_mw": predicted.max(axis=1),
            "point_peak_hour_position": np.argmax(predicted, axis=1) + 1,
        }
    )
    base["days_since_previous_event"] = base["target_date"].diff().dt.days
    base["episode_id"] = (base["days_since_previous_event"].fillna(999).gt(1)).cumsum().astype(int)
    threshold_values = selected.groupby("target_date")["threshold_mw"].first()
    base["frozen_pre_validation_threshold_mw"] = base["target_date"].map(threshold_values)
    for method in methods:
        frame = selected.loc[selected["method"] == method].set_index("target_date")
        safe = method.replace("validation_quantile_point_", "vq_point_").replace(
            "equal_alert_point_", "equal_alert_"
        )
        base[f"{safe}_alert"] = base["target_date"].map(frame["alert"]).astype(bool)
        base[f"{safe}_decision_score_mw"] = base["target_date"].map(frame["decision_score_mw"])
        base[f"{safe}_daily_cost"] = base["target_date"].map(frame["daily_cost"])
    episode_summary = (
        base.groupby("episode_id", as_index=False)
        .agg(
            episode_start=("target_date", "min"),
            episode_end=("target_date", "max"),
            peak_event_days=("target_date", "size"),
            maximum_actual_peak_mw=("actual_peak_mw", "max"),
            tuned_point_alerted_events=("tuned_point_alert", "sum"),
            two_sided_alerted_events=("two_sided_conformal_alert", "sum"),
            one_sided_alerted_events=("one_sided_conformal_alert", "sum"),
        )
    )
    cube = decision_cube.loc[
        decision_cube["peak_threshold"].eq(threshold_spec)
        & decision_cube["cost_ratio"].eq(cost_ratio)
        & decision_cube["method"].isin(methods)
    ].copy()
    return base, episode_summary, cube


def threshold_definition_audit(
    iso_daily: pd.DataFrame,
    nyiso_daily: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset, daily in (("ISO_NE", iso_daily), ("NYISO", nyiso_daily)):
        for usage, cutoff in (
            ("decision_pre_validation_frozen", pd.Timestamp("2024-01-01")),
            ("2024_band_diagnostic_year_specific", pd.Timestamp("2024-01-01")),
            ("2025_band_diagnostic_year_specific", pd.Timestamp("2025-01-01")),
            ("2026H1_band_diagnostic_year_specific", pd.Timestamp("2026-01-01")),
        ):
            if cutoff > pd.to_datetime(daily["date"]).max():
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "usage": usage,
                    "definition": "linear 95th percentile of all available daily peaks strictly before cutoff",
                    "cutoff_date": cutoff,
                    "threshold_mw": historical_peak_threshold(daily, cutoff, 95.0),
                }
            )
    return pd.DataFrame(rows)
