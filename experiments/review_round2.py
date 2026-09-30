"""Second-round reviewer experiments on frozen historical out-of-sample forecasts.

Every new analysis is retrospective. This module does not refit the point models
or replace the v1.0 and v1.1 results.
"""

from __future__ import annotations

import csv
import math
import zipfile
from collections import defaultdict
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.revision_common import (
    LOAD_COLUMNS,
    PRED_COLUMNS,
    RESIDUAL_COLUMNS,
    TRUE_COLUMNS,
    _threshold_arrays,
    confusion,
    file_sha256,
    load_yaml,
    tune_parameter,
    write_csv,
    write_json,
    write_parquet,
)


ROOT = Path(__file__).resolve().parents[1]
UPPER_COLUMNS = [f"upper_h{hour:02d}" for hour in range(1, 25)]
LOWER_COLUMNS = [f"lower_h{hour:02d}" for hour in range(1, 25)]
YEAR_2024 = pd.Timestamp("2024-01-01")
YEAR_2025 = pd.Timestamp("2025-01-01")
END_2025 = pd.Timestamp("2025-12-31")


def _rank(n: int, alpha: float) -> int:
    rank = math.ceil((n + 1) * (1 - alpha))
    if not 1 <= rank <= n:
        raise ValueError(f"Non-finite quantile rank {rank} for n={n}, alpha={alpha}")
    return rank


def _sorted_store(path: Path) -> pd.DataFrame:
    result = pd.read_parquet(path).sort_values("target_date").reset_index(drop=True)
    result["target_date"] = pd.to_datetime(result["target_date"])
    if result["target_date"].duplicated().any():
        raise AssertionError(f"Duplicate forecast dates in {path}")
    diffs = result["target_date"].diff().dropna()
    if not (diffs == pd.Timedelta(days=1)).all():
        raise AssertionError(f"Nonconsecutive forecast store: {path}")
    for column in ("training_end", "scaler_fit_end", "model_fit_date"):
        if column in result and not (pd.to_datetime(result[column]) < result["target_date"]).all():
            raise AssertionError(f"Future information in {column}: {path}")
    residual_error = np.max(
        np.abs(result[TRUE_COLUMNS].to_numpy(float) - result[PRED_COLUMNS].to_numpy(float)
               - result[RESIDUAL_COLUMNS].to_numpy(float))
    )
    if residual_error > 1e-6:
        raise AssertionError(f"Stored residual mismatch: {residual_error}")
    return result


def load_inputs(config: dict[str, Any]) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    base = ROOT / "results" / config["base_pipeline_version"]
    data_root = ROOT / "outputs" / "01_data_audit"
    input_paths = {
        "ISO_NE_features": data_root / "ISO_NE_features.parquet",
        "ISO_NE_daily": data_root / "ISO_NE_daily_profile.parquet",
        "NYISO_features": data_root / "NYISO_features.parquet",
        "NYISO_daily": data_root / "NYISO_daily_profile.parquet",
        "feature_manifest": data_root / "feature_manifest.csv",
        "ISO_NE_oos": base / "A_oos_and_audit" / "oos_prediction_store.parquet",
        "NYISO_oos": base / "F_nyiso_external" / "nyiso_oos_prediction_store.parquet",
        "ISO_NE_bands": base / "B_conformal" / "conformal_prediction_store.parquet",
        "NYISO_bands": base / "F_nyiso_external" / "nyiso_conformal_prediction_store.parquet",
    }
    hashes = pd.DataFrame(
        [{"input": key, "path": path.relative_to(ROOT).as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size}
         for key, path in input_paths.items()]
    )
    frozen = load_yaml(ROOT / "configs" / "revision_v1.yaml")["freeze"]
    expected = {
        "ISO_NE_features": frozen["iso_features_sha256"],
        "ISO_NE_daily": frozen["iso_daily_sha256"],
        "feature_manifest": frozen["feature_manifest_sha256"],
    }
    for key, digest in expected.items():
        actual = hashes.loc[hashes.input == key, "sha256"].iloc[0]
        if actual != digest:
            raise AssertionError(f"Frozen {key} hash changed: {actual}")
    sources = {
        name: {
            "store": _sorted_store(input_paths[f"{name}_oos"]),
            "daily": pd.read_parquet(input_paths[f"{name}_daily"]),
            "old_bands": pd.read_parquet(input_paths[f"{name}_bands"]),
        }
        for name in ("ISO_NE", "NYISO")
    }
    expected_primary = {
        "ISO_NE": (310, {"tuned_point": 24, "two_sided_conformal": 17, "one_sided_conformal": 16}),
        "NYISO": (308, {"tuned_point": 11, "two_sided_conformal": 22, "one_sided_conformal": 20}),
    }
    checks = []
    for name, parts in sources.items():
        point = parts["store"].loc[parts["store"].target_date.between(YEAR_2025, END_2025)]
        band = parts["old_bands"].loc[
            (parts["old_bands"].method == "standardized_max_90")
            & pd.to_datetime(parts["old_bands"].target_date).between(YEAR_2025, END_2025)
        ]
        merged = point.merge(band, on=["target_date", "dataset"], validate="one_to_one")
        actual = merged[TRUE_COLUMNS].to_numpy(float)
        covered = int(np.sum(np.all(
            (actual >= merged[LOWER_COLUMNS].to_numpy(float))
            & (actual <= merged[UPPER_COLUMNS].to_numpy(float)), axis=1
        )))
        if len(merged) != 365 or covered != expected_primary[name][0]:
            raise AssertionError(f"Original {name} coverage gate failed: {covered}/{len(merged)}")
        cube_path = (base / "C_decision_and_bootstrap" / "decision_cube_2025.csv" if name == "ISO_NE"
                     else base / "F_nyiso_external" / "nyiso_decision_cube_2025.csv")
        cube = pd.read_csv(cube_path)
        for method, cost in expected_primary[name][1].items():
            selected = cube.loc[(cube.method == method) & (cube.peak_threshold == "historical_q95")
                                & np.isclose(cube.cost_ratio, 10)]
            if len(selected) != 1 or not np.isclose(float(selected.cost.iloc[0]), cost):
                raise AssertionError(f"Original {name} cost gate failed for {method}")
        checks.append({"dataset": name, "days_2025": 365, "joint_covered_days": covered,
                       "status": "PASS"})
    return sources, hashes.assign(status="FROZEN")


def _window_residual(store: pd.DataFrame, index: int, window: int) -> np.ndarray:
    if index < window:
        raise AssertionError("Insufficient residual history")
    dates = store["target_date"]
    if dates.iloc[index - window] != dates.iloc[index] - pd.Timedelta(days=window):
        raise AssertionError("Window contains missing or future dates")
    return store.iloc[index - window:index][RESIDUAL_COLUMNS].to_numpy(float)


def _band_record(store: pd.DataFrame, index: int, method: str, family: str, window: int,
                 lower: np.ndarray, upper: np.ndarray, q: float, rank: int,
                 alpha_state: float = math.nan, step_size: float = math.nan,
                 state_at_boundary: bool = False) -> dict[str, Any]:
    target = store.at[index, "target_date"]
    row: dict[str, Any] = {
        "dataset": store.at[index, "dataset"], "target_date": target, "method": method,
        "family": family, "window_days": window, "score_start": target - pd.Timedelta(days=window),
        "score_end": target - pd.Timedelta(days=1), "quantile_rank": rank,
        "quantile_scalar": q, "alpha_state": alpha_state, "step_size": step_size,
        "state_at_boundary": state_at_boundary,
    }
    for hour in range(24):
        row[LOWER_COLUMNS[hour]] = lower[hour]
        row[UPPER_COLUMNS[hour]] = upper[hour]
    return row


def build_rolling_bands(store: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    predicted = store[PRED_COLUMNS].to_numpy(float)
    actual = store[TRUE_COLUMNS].to_numpy(float)
    start = int(store.index[store.target_date == YEAR_2024][0])
    end = int(store.index[store.target_date == END_2025][0])
    rows: list[dict[str, Any]] = []
    floor = float(config["scale_floor_mw"])
    alpha = float(config["alpha"])
    for window in config["calibration_windows_days"]:
        window = int(window)
        rank = _rank(window, alpha)
        for index in range(start, end + 1):
            errors = _window_residual(store, index, window)
            sigma = np.maximum(floor, np.median(np.abs(errors), axis=0))
            for family in ("two_sided", "one_sided"):
                scores = np.max((np.abs(errors) if family == "two_sided" else errors) / sigma, axis=1)
                q = float(np.partition(scores, rank - 1)[rank - 1])
                margin = q * sigma
                lower = predicted[index] - margin if family == "two_sided" else np.full(24, np.nan)
                upper = predicted[index] + margin
                rows.append(_band_record(store, index, f"{family}_w{window}", family,
                                         window, lower, upper, q, rank))
    for eta in config["aci_step_sizes"]:
        eta = float(eta)
        alpha_state = alpha
        window = 90
        low, high = 1 / (window + 1), window / (window + 1)
        for index in range(start, end + 1):
            errors = _window_residual(store, index, window)
            sigma = np.maximum(floor, np.median(np.abs(errors), axis=0))
            rank = _rank(window, alpha_state)
            scores = np.max(np.abs(errors) / sigma, axis=1)
            q = float(np.partition(scores, rank - 1)[rank - 1])
            margin = q * sigma
            lower, upper = predicted[index] - margin, predicted[index] + margin
            rows.append(_band_record(store, index, f"aci_eta{eta:g}", "aci_projected",
                                     window, lower, upper, q, rank, alpha_state, eta,
                                     bool(np.isclose(alpha_state, low) or np.isclose(alpha_state, high))))
            miss = float(np.any((actual[index] < lower) | (actual[index] > upper)))
            alpha_state = float(np.clip(alpha_state + eta * (alpha - miss), low, high))
    result = pd.DataFrame(rows)
    if not (pd.to_datetime(result.score_end) < pd.to_datetime(result.target_date)).all():
        raise AssertionError("Future residual used in new band")
    return result


def build_peak_bands(store: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    actual_peak = store[TRUE_COLUMNS].to_numpy(float).max(axis=1)
    predicted_peak = store[PRED_COLUMNS].to_numpy(float).max(axis=1)
    peak_errors = actual_peak - predicted_peak
    start = int(store.index[store.target_date == YEAR_2024][0])
    end = int(store.index[store.target_date == END_2025][0])
    rows = []
    for window in config["peak_calibration_windows_days"]:
        window = int(window)
        rank = _rank(window, float(config["alpha"]))
        for index in range(start, end + 1):
            _window_residual(store, index, window)
            q = float(np.partition(peak_errors[index - window:index], rank - 1)[rank - 1])
            rows.append({"dataset": store.at[index, "dataset"], "target_date": store.at[index, "target_date"],
                         "method": f"direct_peak_w{window}", "family": "direct_peak",
                         "window_days": window, "score_end": store.at[index, "target_date"] - pd.Timedelta(days=1),
                         "quantile_rank": rank, "quantile_scalar": q,
                         "actual_peak_mw": actual_peak[index], "predicted_peak_mw": predicted_peak[index],
                         "upper_peak_mw": predicted_peak[index] + q,
                         "upper_covered": bool(actual_peak[index] <= predicted_peak[index] + q)})
    return pd.DataFrame(rows)


def band_quality(store: pd.DataFrame, bands: pd.DataFrame, peak_bands: pd.DataFrame,
                 alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = bands.merge(store[["target_date", "dataset", *TRUE_COLUMNS, *PRED_COLUMNS]],
                         on=["target_date", "dataset"], validate="many_to_one")
    actual = merged[TRUE_COLUMNS].to_numpy(float)
    predicted = merged[PRED_COLUMNS].to_numpy(float)
    upper = merged[UPPER_COLUMNS].to_numpy(float)
    lower = merged[LOWER_COLUMNS].to_numpy(float)
    both = np.isfinite(lower).all(axis=1)
    joint = np.full(len(merged), np.nan)
    joint[both] = np.all((actual[both] >= lower[both]) & (actual[both] <= upper[both]), axis=1)
    upper_ok = np.all(actual <= upper, axis=1)
    width = np.full(len(merged), np.nan)
    width[both] = np.mean(upper[both] - lower[both], axis=1)
    interval_score = np.full(len(merged), np.nan)
    interval_score[both] = np.mean(
        upper[both] - lower[both]
        + (2 / alpha) * np.maximum(lower[both] - actual[both], 0)
        + (2 / alpha) * np.maximum(actual[both] - upper[both], 0), axis=1
    )
    pinball = np.mean((0.9 - (actual < upper).astype(float)) * (actual - upper), axis=1)
    daily = merged[["dataset", "target_date", "method", "family", "window_days",
                    "step_size", "alpha_state", "state_at_boundary"]].copy()
    daily["joint_covered"] = joint
    daily["upper_covered"] = upper_ok
    daily["mean_upper_margin_mw"] = np.mean(upper - predicted, axis=1)
    daily["mean_width_mw"] = width
    daily["interval_score_mw"] = interval_score
    daily["upper_pinball_mw"] = pinball
    daily["period"] = pd.to_datetime(daily.target_date).dt.year
    metrics = daily.groupby(["dataset", "method", "family", "window_days", "period"],
                            dropna=False).agg(
        days=("target_date", "size"), joint_coverage=("joint_covered", "mean"),
        upper_coverage=("upper_covered", "mean"), mean_width_mw=("mean_width_mw", "mean"),
        interval_score_mw=("interval_score_mw", "mean"),
        upper_margin_mw=("mean_upper_margin_mw", "mean"),
        upper_pinball_mw=("upper_pinball_mw", "mean"),
        boundary_days=("state_at_boundary", "sum"),
    ).reset_index()
    peak = peak_bands.copy()
    peak["period"] = pd.to_datetime(peak.target_date).dt.year
    peak_metrics = peak.groupby(["dataset", "method", "window_days", "period"]).agg(
        days=("target_date", "size"), peak_upper_coverage=("upper_covered", "mean"),
        mean_peak_margin_mw=("quantile_scalar", "mean"),
    ).reset_index()
    return daily, pd.concat([metrics, peak_metrics], ignore_index=True)


def _tie_audit(event: np.ndarray, predicted_peak: np.ndarray, threshold: np.ndarray,
               delta_grid: np.ndarray, ratio: float) -> dict[str, Any]:
    choices = []
    for delta in delta_grid:
        met = confusion(event, predicted_peak + delta > threshold, ratio)
        choices.append((met["cost"], met["FN"], met["FP"], float(delta)))
    implemented = min(choices, key=lambda x: (x[0], x[1], x[2], abs(x[3]), x[3]))
    stated = min(choices, key=lambda x: (x[0], x[1], x[2], x[3]))
    return {"implemented_delta_mw": implemented[3], "stated_delta_mw": stated[3],
            "tie_rule_changes_choice": implemented[3] != stated[3],
            "validation_cost": implemented[0]}


def decision_grid(name: str, store: pd.DataFrame, daily: pd.DataFrame, bands: pd.DataFrame,
                  peak_bands: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val = store.loc[store.target_date.between(YEAR_2024, YEAR_2025 - pd.Timedelta(days=1))]
    test = store.loc[store.target_date.between(YEAR_2025, END_2025)]
    if len(val) != 366 or len(test) != 365:
        raise AssertionError(f"{name}: unexpected validation/test lengths")
    val_dates = val.target_date.reset_index(drop=True)
    test_dates = test.target_date.reset_index(drop=True)
    actual_val = val[TRUE_COLUMNS].to_numpy(float).max(axis=1)
    actual_test = test[TRUE_COLUMNS].to_numpy(float).max(axis=1)
    pred_val = val[PRED_COLUMNS].to_numpy(float)
    pred_test = test[PRED_COLUMNS].to_numpy(float)
    peak_val = pred_val.max(axis=1)
    peak_test = pred_test.max(axis=1)
    delta_grid = np.arange(float(config["delta_start_mw"]),
                           float(config["delta_end_mw"]) + 0.5 * float(config["delta_step_mw"]),
                           float(config["delta_step_mw"]))
    lambda_grid = [float(x) for x in config["lambda_grid"]]
    threshold_specs = [f"historical_q{x:g}" for x in config["peak_percentiles"]]
    if config["include_rolling_365d_95"]:
        threshold_specs.append("rolling_365d_95")
    score_sources: dict[str, tuple[str, int, float, Any]] = {}
    for method, frame in bands.groupby("method", sort=True):
        band_val = frame.loc[frame.target_date.between(YEAR_2024, YEAR_2025 - pd.Timedelta(days=1))]
        band_test = frame.loc[frame.target_date.between(YEAR_2025, END_2025)]
        band_val = band_val.sort_values("target_date")
        band_test = band_test.sort_values("target_date")
        if not np.array_equal(band_val.target_date.to_numpy(), val_dates.to_numpy()) or not np.array_equal(
            band_test.target_date.to_numpy(), test_dates.to_numpy()
        ):
            raise AssertionError(f"{name} {method}: band date mismatch")
        margins_val = band_val[UPPER_COLUMNS].to_numpy(float) - pred_val
        margins_test = band_test[UPPER_COLUMNS].to_numpy(float) - pred_test
        score_sources[method] = (
            str(frame.family.iloc[0]), int(frame.window_days.iloc[0]), float(frame.step_size.iloc[0]),
            lambda value, pv=pred_val, pt=pred_test, mv=margins_val, mt=margins_test:
                (np.max(pv + value * mv, axis=1), np.max(pt + value * mt, axis=1)),
        )
    for method, frame in peak_bands.groupby("method", sort=True):
        peak_band_val = frame.loc[frame.target_date.between(YEAR_2024, YEAR_2025 - pd.Timedelta(days=1))].sort_values("target_date")
        peak_band_test = frame.loc[frame.target_date.between(YEAR_2025, END_2025)].sort_values("target_date")
        if not np.array_equal(peak_band_val.target_date.to_numpy(), val_dates.to_numpy()) or not np.array_equal(
            peak_band_test.target_date.to_numpy(), test_dates.to_numpy()
        ):
            raise AssertionError(f"{name} {method}: peak-band date mismatch")
        q_val = peak_band_val.quantile_scalar.to_numpy(float)
        q_test = peak_band_test.quantile_scalar.to_numpy(float)
        score_sources[method] = (
            "direct_peak", int(frame.window_days.iloc[0]), math.nan,
            lambda value, pv=peak_val, pt=peak_test, qv=q_val, qt=q_test:
                (pv + value * qv, pt + value * qt),
        )
    summary: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    for spec in threshold_specs:
        threshold_val, threshold_test, fixed = _threshold_arrays(
            daily, val_dates, test_dates, spec, YEAR_2024
        )
        event_val = actual_val > threshold_val
        event_test = actual_test > threshold_test
        for ratio_raw in config["cost_ratios"]:
            ratio = float(ratio_raw)
            delta, point_val_metrics = tune_parameter(
                event_val, lambda shift: peak_val + shift > threshold_val, delta_grid, ratio
            )
            point_alert_test = peak_test + delta > threshold_test
            point_test_metrics = confusion(event_test, point_alert_test, ratio)
            audit = _tie_audit(event_val, peak_val, threshold_val, delta_grid, ratio)
            stated_test = confusion(event_test, peak_test + audit["stated_delta_mw"] > threshold_test, ratio)
            tie_rows.append({"dataset": name, "threshold_spec": spec, "cost_ratio": ratio,
                             **audit, "implemented_test_cost": point_test_metrics["cost"],
                             "stated_rule_test_cost": stated_test["cost"]})
            common = {"dataset": name, "threshold_spec": spec, "fixed_threshold_mw": fixed,
                      "cost_ratio": ratio, "validation_events": int(event_val.sum()),
                      "evaluation_events": int(event_test.sum())}
            summary.append({**common, "method": "tuned_point", "family": "tuned_point",
                            "window_days": math.nan, "step_size": math.nan,
                            "selected_parameter": delta, "parameter_type": "delta_mw",
                            **{f"validation_{key}": value for key, value in point_val_metrics.items()},
                            **{f"test_{key}": value for key, value in point_test_metrics.items()}})
            for i, target in enumerate(test_dates):
                daily_rows.append({"dataset": name, "target_date": target,
                                   "threshold_spec": spec, "cost_ratio": ratio,
                                   "method": "tuned_point", "family": "tuned_point",
                                   "window_days": math.nan, "selected_parameter": delta,
                                   "event": bool(event_test[i]), "alert": bool(point_alert_test[i]),
                                   "daily_loss": int(point_alert_test[i] and not event_test[i])
                                   + ratio * int(event_test[i] and not point_alert_test[i]),
                                   "actual_peak_mw": actual_test[i], "predicted_peak_mw": peak_test[i],
                                   "threshold_mw": threshold_test[i], "decision_score_mw": peak_test[i] + delta})
            for method, (family, window, eta, scorer) in score_sources.items():
                chosen, met_val = tune_parameter(
                    event_val, lambda value, sc=scorer: sc(value)[0] > threshold_val,
                    lambda_grid, ratio,
                )
                score_val, score_test = scorer(chosen)
                alert_test = score_test > threshold_test
                met_test = confusion(event_test, alert_test, ratio)
                summary.append({**common, "method": method, "family": family, "window_days": window,
                                "step_size": eta, "selected_parameter": chosen, "parameter_type": "lambda",
                                **{f"validation_{key}": value for key, value in met_val.items()},
                                **{f"test_{key}": value for key, value in met_test.items()}})
                for i, target in enumerate(test_dates):
                    daily_rows.append({"dataset": name, "target_date": target,
                                       "threshold_spec": spec, "cost_ratio": ratio,
                                       "method": method, "family": family,
                                       "window_days": window, "selected_parameter": chosen,
                                       "event": bool(event_test[i]), "alert": bool(alert_test[i]),
                                       "daily_loss": int(alert_test[i] and not event_test[i])
                                       + ratio * int(event_test[i] and not alert_test[i]),
                                       "actual_peak_mw": actual_test[i], "predicted_peak_mw": peak_test[i],
                                       "threshold_mw": threshold_test[i], "decision_score_mw": score_test[i]})
    return pd.DataFrame(summary), pd.DataFrame(daily_rows), pd.DataFrame(tie_rows)


def select_retrospective_2024(summary: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    primary = summary.loc[(summary.threshold_spec == "historical_q95")
                          & np.isclose(summary.cost_ratio, 10)].copy()
    rows = []
    for (dataset, family), frame in primary.loc[primary.family.isin(["two_sided", "one_sided"])].groupby(
        ["dataset", "family"]
    ):
        ordered = frame.assign(abs_parameter=frame.selected_parameter.abs()).sort_values(
            ["validation_cost", "validation_FN", "validation_FP", "abs_parameter",
             "selected_parameter", "window_days"]
        )
        selected = ordered.iloc[0].to_dict()
        selected["selection_policy"] = "retrospective_2024_joint_window_and_lambda_cost_minimization"
        rows.append(selected)
    for dataset, frame in quality.loc[(quality.family == "aci_projected") & (quality.period == 2024)].groupby("dataset"):
        ordered = frame.assign(coverage_gap=(frame.joint_coverage - 0.9).abs()).sort_values(
            ["coverage_gap", "mean_width_mw", "method"]
        )
        selected = ordered.iloc[0]
        rows.append({"dataset": dataset, "family": "aci_projected", "method": selected.method,
                     "window_days": 90, "step_size": float(selected.method.removeprefix("aci_eta")),
                     "validation_joint_coverage": selected.joint_coverage,
                     "validation_mean_width_mw": selected.mean_width_mw,
                     "selection_policy": "retrospective_2024_absolute_joint_coverage_gap_then_width"})
    return pd.DataFrame(rows)


def _resample_indices(n: int, block: int, replicates: int, seed: int,
                      design: str, quarters: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(seed)

    def sample_segment(length: int, circular: bool) -> np.ndarray:
        count = math.ceil(length / block)
        maximum = length if circular else length - block + 1
        if maximum <= 0:
            raise ValueError("Block exceeds calendar segment")
        starts = rng.integers(0, maximum, size=(replicates, count), dtype=np.int32)
        indexes = starts[:, :, None] + np.arange(block, dtype=np.int32)[None, None, :]
        if circular:
            indexes %= length
        return indexes.reshape(replicates, -1)[:, :length]

    if design == "annual_circular":
        return sample_segment(n, True)
    if design == "annual_non_circular":
        return sample_segment(n, False)
    if design == "quarter_stratified_non_circular":
        pieces = []
        for quarter in np.unique(quarters):
            positions = np.flatnonzero(quarters == quarter)
            if not np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
                raise AssertionError("Noncontiguous calendar quarter")
            pieces.append(sample_segment(len(positions), False) + positions[0])
        return np.concatenate(pieces, axis=1)
    raise ValueError(design)


def seasonal_diagnostics(name: str, quality_daily: pd.DataFrame, decision_daily: pd.DataFrame,
                         config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                         pd.DataFrame, pd.DataFrame]:
    point = decision_daily.loc[decision_daily.method == "tuned_point"].sort_values("target_date")
    if len(point) != 365:
        raise AssertionError(f"{name}: missing primary point-rule days")
    dates = pd.to_datetime(point.target_date).reset_index(drop=True)
    events = point.event.to_numpy(bool)
    point_loss = point.daily_loss.to_numpy(float)
    quarters = dates.dt.quarter.to_numpy(int)
    monthly_rows = []
    episode_rows = []
    bootstrap_rows = []
    acf_rows = []
    all_daily = []
    for method in ("two_sided_w90", "one_sided_w90"):
        chosen = decision_daily.loc[decision_daily.method == method].sort_values("target_date")
        quality = quality_daily.loc[quality_daily.method == method].sort_values("target_date")
        if len(chosen) != 365 or len(quality) != 365 or not np.array_equal(
            chosen.target_date.to_numpy(), point.target_date.to_numpy()
        ) or not np.array_equal(quality.target_date.to_numpy(), point.target_date.to_numpy()):
            raise AssertionError(f"{name}: daily dates do not align for {method}")
        alert = chosen.alert.to_numpy(bool)
        loss = chosen.daily_loss.to_numpy(float)
        delta = loss - point_loss
        upper = quality.upper_covered.to_numpy(bool)
        joint = quality.joint_covered.to_numpy(float)
        frame = pd.DataFrame({"dataset": name, "target_date": dates, "month": dates.dt.month,
                              "quarter": quarters, "method": method, "event": events,
                              "point_alert": point.alert.to_numpy(bool), "band_alert": alert,
                              "point_loss": point_loss, "band_loss": loss, "paired_loss": delta,
                              "joint_covered": joint, "upper_covered": upper})
        all_daily.append(frame)
        for month, group in frame.groupby("month"):
            monthly_rows.append({"dataset": name, "month": int(month), "method": method,
                                 "days": len(group), "events": int(group.event.sum()),
                                 "joint_covered_days": (int(group.joint_covered.sum()) if method == "two_sided_w90"
                                                        else math.nan),
                                 "upper_covered_days": int(group.upper_covered.sum()),
                                 "point_cost": float(group.point_loss.sum()),
                                 "band_cost": float(group.band_loss.sum()),
                                 "paired_cost_difference": float(group.paired_loss.sum()),
                                 "band_TP": int((group.event & group.band_alert).sum()),
                                 "band_FP": int((~group.event & group.band_alert).sum()),
                                 "band_FN": int((group.event & ~group.band_alert).sum())})
        episode_number = -1
        in_episode = False
        for i, event in enumerate(events):
            if event and not in_episode:
                episode_number += 1
                in_episode = True
            if not event:
                in_episode = False
            if event:
                frame.loc[i, "episode_id"] = episode_number
        for number, group in frame.dropna(subset=["episode_id"]).groupby("episode_id"):
            idx = group.index.to_numpy(int)
            episode_rows.append({"dataset": name, "method": method, "episode_id": int(number),
                                 "start": dates.iloc[idx[0]], "end": dates.iloc[idx[-1]],
                                 "event_days": len(idx), "episode_paired_cost": float(delta[idx].sum()),
                                 "annual_paired_cost": float(delta.sum()),
                                 "annual_difference_without_episode": float(delta.sum() - delta[idx].sum())})
        series = {"paired_daily_loss": delta, "upper_failure": (~upper).astype(float)}
        if method == "two_sided_w90":
            series["joint_failure"] = 1 - joint
        for label, values in series.items():
            for lag in range(1, 29):
                if np.std(values[:-lag]) == 0 or np.std(values[lag:]) == 0:
                    autocorrelation = math.nan
                else:
                    autocorrelation = float(np.corrcoef(values[:-lag], values[lag:])[0, 1])
                acf_rows.append({"dataset": name, "method": method, "series": label,
                                 "lag_days": lag, "autocorrelation": autocorrelation})
        for block in config["bootstrap_blocks_days"]:
            block = int(block)
            for design_index, design in enumerate(("annual_circular", "annual_non_circular",
                                                    "quarter_stratified_non_circular")):
                indices = _resample_indices(
                    365, block, int(config["bootstrap_replicates"]),
                    int(config["random_seed"]) + (0 if name == "ISO_NE" else 10000)
                    + (0 if method == "two_sided_w90" else 1000) + block * 10 + design_index,
                    design, quarters,
                )
                targets = {"paired_annual_cost_difference": (delta, "sum"),
                           "daily_upper_coverage": (upper.astype(float), "mean")}
                if method == "two_sided_w90":
                    targets["daily_joint_coverage"] = (joint, "mean")
                event_counts = events[indices].sum(axis=1)
                for metric, (values, operation) in targets.items():
                    sampled = values[indices].sum(axis=1) if operation == "sum" else values[indices].mean(axis=1)
                    bootstrap_rows.append({"dataset": name, "method": method, "metric": metric,
                                           "design": design, "block_days": block,
                                           "replicates": len(sampled),
                                           "observed": float(values.sum() if operation == "sum" else values.mean()),
                                           "resample_p025": float(np.quantile(sampled, 0.025)),
                                           "resample_p975": float(np.quantile(sampled, 0.975)),
                                           "resampled_events_p025": float(np.quantile(event_counts, 0.025)),
                                           "resampled_events_p975": float(np.quantile(event_counts, 0.975)),
                                           "interpretation": "conditional_descriptive_resampling_not_future_year_CI"})
    return (pd.DataFrame(monthly_rows), pd.concat(all_daily, ignore_index=True),
            pd.DataFrame(episode_rows), pd.DataFrame(bootstrap_rows), pd.DataFrame(acf_rows))


def nyiso_target_audit(decision_daily: pd.DataFrame, downloads: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit timestamps in user-supplied official NYISO monthly archives."""
    irregular: dict[str, set[tuple[str, str]]] = defaultdict(set)
    raw_files = []
    for month in range(1, 13):
        path = downloads / f"2025{month:02d}01pal_csv.zip"
        if not path.is_file():
            raise FileNotFoundError(path)
        raw_files.append(path.name)
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                with archive.open(member) as raw, TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
                    for row in csv.DictReader(stream):
                        stamp = row["Time Stamp"]
                        if int(stamp[14:16]) % 5 or int(stamp[17:19]):
                            day = datetime.strptime(stamp[:10], "%m/%d/%Y").strftime("%Y-%m-%d")
                            irregular[day].add((stamp, row["Time Zone"]))
    audit_root = ROOT / "outputs" / "01_data_audit"
    imputed = pd.read_csv(audit_root / "imputation_log_nyiso.csv")
    imputed_days = set(imputed.loc[imputed.timestamp_local.astype(str).str.startswith("2025"),
                                   "timestamp_local"].astype(str).str[:10])
    dst = pd.read_csv(audit_root / "dst_adjustments_nyiso.csv")
    dst_days = set(dst.loc[dst.date.astype(str).str.startswith("2025"), "date"].astype(str))
    events = decision_daily.loc[decision_daily.method == "tuned_point"].copy()
    events["day"] = pd.to_datetime(events.target_date).dt.strftime("%Y-%m-%d")
    rows = []
    for row in events.itertuples(index=False):
        rows.append({"dataset": "NYISO", "date": row.day, "peak_event": bool(row.event),
                     "irregular_timestamp_count": len(irregular.get(row.day, set())),
                     "irregular_day": row.day in irregular,
                     "interpolated_day": row.day in imputed_days,
                     "dst_adjusted_day": row.day in dst_days})
    daily = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "dataset": "NYISO", "test_days": len(daily), "peak_events": int(daily.peak_event.sum()),
        "irregular_days": int(daily.irregular_day.sum()),
        "irregular_peak_days": int((daily.irregular_day & daily.peak_event).sum()),
        "irregular_unique_timestamps": int(daily.irregular_timestamp_count.sum()),
        "interpolated_days": int(daily.interpolated_day.sum()),
        "interpolated_peak_days": int((daily.interpolated_day & daily.peak_event).sum()),
        "dst_adjusted_days": int(daily.dst_adjusted_day.sum()),
        "dst_adjusted_peak_days": int((daily.dst_adjusted_day & daily.peak_event).sum()),
        "raw_archives": "|".join(raw_files),
    }])
    return daily, summary
