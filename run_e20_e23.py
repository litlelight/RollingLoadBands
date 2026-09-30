from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / "work" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

from experiments.e20_e23_closure import (
    band_metrics_for_periods,
    build_rolling_marginal_band,
    build_rolling_raw_max_band,
    build_rolling_standardized_bands,
    build_static_previous_year_band,
    coverage_bootstrap,
    calibration_window_selection_audit,
    development_calibration_sensitivity,
    peak_event_audit,
    ridge_development_sensitivity,
    threshold_definition_audit,
)
from experiments.revision_common import (
    file_sha256,
    hardware_snapshot,
    load_feature_inputs,
    load_yaml,
    log,
    output_hash_manifest,
    software_snapshot,
    write_csv,
    write_json,
    write_parquet,
)


def _stage(name: str, started: float, status: str = "PASS") -> dict[str, Any]:
    return {"stage": name, "status": status, "runtime_seconds": time.perf_counter() - started}


def _md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in selected.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_experiment_report(result_root: Path, report_root: Path, config: dict[str, Any]) -> Path:
    e20 = pd.read_csv(result_root / "E20_coverage_bootstrap" / "coverage_moving_block_bootstrap.csv")
    e21 = pd.read_csv(result_root / "E21_band_robustness" / "band_method_metrics.csv")
    window_selection = pd.read_csv(
        result_root / "E21_band_robustness" / "calibration_window_selection_2024.csv"
    )
    ridge = pd.read_csv(result_root / "E22_development_sensitivity" / "ridge_parameter_metrics.csv")
    windows = pd.read_csv(result_root / "E22_development_sensitivity" / "calibration_window_metrics.csv")
    events = pd.read_csv(result_root / "E23_peak_event_audit" / "peak_event_audit_2025.csv")
    primary = e20.loc[
        (e20["dataset"] == "ISO_NE")
        & (e20["period"].astype(str) == "2025")
        & (e20["block_length_days"] == int(config["coverage_bootstrap"]["primary_block_days"]))
    ]
    methods_2025 = e21.loc[e21["period"].astype(str) == "2025"]
    ridge_overall = ridge.loc[ridge["period"] == "2020-2023"].sort_values("mae_mw")
    window_overall = windows.loc[windows["period"] == "2020-2023"].sort_values("calibration_window_days")
    lines = [
        f"# {config['pipeline_version']} E20–E23 实验报告",
        "",
        "本文件只记录新增实验结果，不包含论文改写、审稿回复或投稿材料。所有实验沿用已通过 E2/E3 门禁的 OOS forecast store。",
        "",
        "## E20 Coverage moving-block bootstrap",
        "",
        _md_table(primary, ["method", "metric", "observed_coverage_pct", "block_length_days", "ci_lower_pct", "ci_upper_pct", "bootstrap_fraction_below_nominal"]),
        "",
        "完整文件同时包含 ISO-NE 2024/2025/2026H1 和 NYISO 2024/2025，以及 block=3/7/14/28 天。",
        "",
        "## E21 Band robustness",
        "",
        _md_table(methods_2025, ["method", "daily_simultaneous_coverage_pct", "daily_upper_coverage_pct", "peak_hour_coverage_pct", "peak_hour_upper_coverage_pct", "mean_upper_margin_mw", "mean_width_mw", "interval_score_mw"]),
        "",
        "另行保存 calibration-window 30/60/90/180/270/365 天与 scale-floor 25/50/100 MW 的完整逐日 bands 和指标。",
        "",
        "### 2024 calibration-window selection audit",
        "",
        _md_table(
            window_selection,
            [
                "calibration_window_days",
                "daily_simultaneous_coverage_pct",
                "absolute_daily_coverage_gap_pp",
                "mean_width_mw",
                "interval_score_mw",
                "rank_coverage_closeness",
                "rank_mean_width",
                "rank_interval_score",
                "pareto_dominated_on_three_reported_metrics",
                "originally_frozen_window",
            ],
        ),
        "",
        "归档实验记录无法恢复当时的单一 objective/selection criterion；90 天仅按原冻结设置保留，不把 2024 或 2025 的事后最优值重新定义为主配置。",
        "",
        "## E22 Development-period sensitivity",
        "",
        "### Ridge α × training window（2020–2023 only）",
        "",
        _md_table(ridge_overall, ["configuration", "ridge_alpha", "training_window_years", "frozen_primary", "mae_mw", "rmse_mw", "peak_mae_mw"]),
        "",
        "### Calibration window（2020–2023 only）",
        "",
        _md_table(window_overall, ["calibration_window_days", "frozen_primary", "daily_simultaneous_coverage_pct", "mean_width_mw", "interval_score_mw", "absolute_daily_coverage_error_pp"]),
        "",
        "这些表是开发期敏感性，不使用 2025 结果重新选择主配置。",
        "",
        "## E23 2025 peak-event audit",
        "",
        f"共 {len(events)} 个事件日，按 pre-validation frozen historical Q95 threshold 定义。逐事件 alert、decision score 和 daily cost 均保存在 CSV。",
        "",
    ]
    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / "E20_E23_EXPERIMENT_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E20-E23 final reviewer-closure experiments only.")
    parser.add_argument("--config", default="configs/reviewer_closure_v1.yaml")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_yaml(config_path)
    version = str(config["pipeline_version"])
    base_root = ROOT / "results" / str(config["base_pipeline_version"])
    result_root = ROOT / "results" / version
    report_root = ROOT / "reports" / version
    e20_dir = result_root / "E20_coverage_bootstrap"
    e21_dir = result_root / "E21_band_robustness"
    e22_dir = result_root / "E22_development_sensitivity"
    e23_dir = result_root / "E23_peak_event_audit"
    audit_dir = result_root / "audit"
    for path in (e20_dir, e21_dir, e22_dir, e23_dir, audit_dir, report_root):
        path.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    timings: list[dict[str, Any]] = []
    write_json(hardware_snapshot(), audit_dir / "hardware.json")
    write_json(software_snapshot(), audit_dir / "software_versions.json")
    shutil.copy2(config_path, audit_dir / "reviewer_closure_v1.yaml")
    try:
        e0_dir = ROOT / "outputs" / "01_data_audit"
        iso_features, iso_daily, manifest = load_feature_inputs(e0_dir, "ISO_NE")
        _, ny_daily, _ = load_feature_inputs(e0_dir, "NYISO")
        iso_store = pd.read_parquet(base_root / "A_oos_and_audit" / "oos_prediction_store.parquet")
        iso_bands = pd.read_parquet(base_root / "B_conformal" / "conformal_prediction_store.parquet")
        ny_store = pd.read_parquet(base_root / "F_nyiso_external" / "nyiso_oos_prediction_store.parquet")
        ny_bands = pd.read_parquet(base_root / "F_nyiso_external" / "nyiso_conformal_prediction_store.parquet")

        started = time.perf_counter()
        log("E20: coverage moving-block bootstrap for ISO-NE and NYISO")
        method_metrics = {
            "standardized_max_90": ["daily_simultaneous_coverage", "daily_upper_coverage"],
            "one_sided_max_90": ["daily_upper_coverage"],
            "bonferroni_365": ["daily_simultaneous_coverage"],
            "disjoint_standardized_max_90": ["daily_simultaneous_coverage"],
        }
        iso_periods = [
            ("2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
            ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
            ("2026H1", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-30")),
        ]
        ny_periods = [
            ("2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
            ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ]
        e20 = pd.concat(
            [
                coverage_bootstrap(iso_store, iso_bands, "ISO_NE", iso_periods, method_metrics, config["coverage_bootstrap"], int(config["random_seed"])),
                coverage_bootstrap(ny_store, ny_bands, "NYISO", ny_periods, method_metrics, config["coverage_bootstrap"], int(config["random_seed"]) + 100_000),
            ],
            ignore_index=True,
        )
        write_csv(e20, e20_dir / "coverage_moving_block_bootstrap.csv")
        timings.append(_stage("E20 coverage bootstrap", started))

        started = time.perf_counter()
        log("E21: static, raw-max, marginal, calibration-window, and scale-floor robustness")
        static = build_static_previous_year_band(iso_store, 50.0, 0.10)
        raw = build_rolling_raw_max_band(iso_store, 90, 0.10)
        marginal = build_rolling_marginal_band(iso_store, 90, 0.10)
        additional = pd.concat([static, raw, marginal], ignore_index=True)
        all_methods = pd.concat([iso_bands, additional], ignore_index=True)
        comparison_metrics = band_metrics_for_periods(iso_store, all_methods, iso_daily, iso_periods)
        write_parquet(additional, e21_dir / "additional_band_predictions.parquet")
        write_csv(comparison_metrics, e21_dir / "band_method_metrics.csv")
        upper_margin_metrics = comparison_metrics[
            [
                "period",
                "method",
                "days",
                "daily_upper_coverage_pct",
                "peak_hour_upper_coverage_pct",
                "mean_upper_margin_mw",
                "peak_threshold_definition",
            ]
        ].copy()
        upper_margin_metrics.insert(0, "dataset", "ISO_NE")
        write_csv(upper_margin_metrics, e21_dir / "upper_margin_metrics.csv")

        window_bands = []
        for window in config["band_robustness"]["calibration_windows_days"]:
            window_bands.append(
                build_rolling_standardized_bands(
                    iso_store,
                    int(window),
                    float(config["band_robustness"]["scale_floor_mw"]),
                    float(config["band_robustness"]["alpha"]),
                    f"standardized_max_w{int(window)}_floor50",
                )
            )
        window_band_frame = pd.concat(window_bands, ignore_index=True)
        window_metrics = band_metrics_for_periods(iso_store, window_band_frame, iso_daily, iso_periods)
        window_metrics["calibration_window_days"] = window_metrics["method"].str.extract(r"w(\d+)").astype(int)
        window_metrics["frozen_primary"] = window_metrics["calibration_window_days"].eq(90)
        write_parquet(window_band_frame, e21_dir / "calibration_window_band_predictions.parquet")
        write_csv(window_metrics, e21_dir / "calibration_window_metrics.csv")
        selection_2024 = calibration_window_selection_audit(
            window_metrics,
            validation_period="2024",
            nominal_coverage_pct=90.0,
            originally_frozen_window_days=90,
        )
        write_csv(selection_2024, e21_dir / "calibration_window_selection_2024.csv")

        floor_bands = []
        for floor in config["band_robustness"]["scale_floors_mw"]:
            floor_bands.append(
                build_rolling_standardized_bands(
                    iso_store,
                    int(config["band_robustness"]["primary_window_days"]),
                    float(floor),
                    float(config["band_robustness"]["alpha"]),
                    f"standardized_max_w90_floor{float(floor):g}",
                )
            )
        floor_band_frame = pd.concat(floor_bands, ignore_index=True)
        floor_metrics = band_metrics_for_periods(iso_store, floor_band_frame, iso_daily, iso_periods)
        floor_metrics["scale_floor_mw_setting"] = floor_metrics["method"].str.extract(r"floor([0-9.]+)").astype(float)
        floor_metrics["frozen_primary"] = floor_metrics["scale_floor_mw_setting"].eq(50.0)
        write_parquet(floor_band_frame, e21_dir / "scale_floor_band_predictions.parquet")
        write_csv(floor_metrics, e21_dir / "scale_floor_metrics.csv")
        threshold_audit = threshold_definition_audit(iso_daily, ny_daily)
        write_csv(threshold_audit, e21_dir / "historical_threshold_definition_audit.csv")
        timings.append(_stage("E21 band robustness", started))

        started = time.perf_counter()
        log("E22: development-only Ridge alpha/training-window and calibration-window sensitivity")
        ridge_predictions, ridge_audit, ridge_metrics = ridge_development_sensitivity(
            iso_features,
            manifest,
            config["development_sensitivity"],
            int(config["hardware"]["cpu_workers"]),
        )
        write_parquet(ridge_predictions, e22_dir / "ridge_parameter_predictions_2020_2023.parquet")
        write_csv(ridge_audit, e22_dir / "ridge_parameter_monthly_fit_audit.csv")
        write_csv(ridge_metrics, e22_dir / "ridge_parameter_metrics.csv")
        development_bands, development_metrics = development_calibration_sensitivity(
            iso_store,
            iso_daily,
            config["development_sensitivity"],
            float(config["band_robustness"]["alpha"]),
            float(config["band_robustness"]["scale_floor_mw"]),
        )
        write_parquet(development_bands, e22_dir / "calibration_window_predictions_2020_2023.parquet")
        write_csv(development_metrics, e22_dir / "calibration_window_metrics.csv")
        timings.append(_stage("E22 development sensitivity", started))

        started = time.perf_counter()
        log("E23: 17-event ISO-NE peak audit")
        decision_daily = pd.read_parquet(base_root / "C_decision_and_bootstrap" / "decision_daily_2025.parquet")
        decision_cube = pd.read_csv(base_root / "C_decision_and_bootstrap" / "decision_cube_2025.csv")
        events, episodes, rule_summary = peak_event_audit(
            iso_store, decision_daily, decision_cube, config["peak_event_audit"]
        )
        write_csv(events, e23_dir / "peak_event_audit_2025.csv")
        write_csv(episodes, e23_dir / "peak_event_episode_summary_2025.csv")
        write_csv(rule_summary, e23_dir / "decision_rule_summary_q95_cost10.csv")
        timings.append(_stage("E23 peak-event audit", started))

        report_path = build_experiment_report(result_root, report_root, config)
        timings.append(
            {
                "stage": "total wall time",
                "status": "PASS",
                "runtime_seconds": time.perf_counter() - started_all,
            }
        )
        write_csv(pd.DataFrame(timings), audit_dir / "runtime.csv")
        write_json(
            {
                "status": "PASS",
                "pipeline_version": version,
                "base_pipeline_version": config["base_pipeline_version"],
                "scope": "E20-E23 experiments only",
                "report": str(report_path),
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            audit_dir / "pipeline_status.json",
        )
        hashes = output_hash_manifest(result_root)
        write_csv(hashes, audit_dir / "results_hashes.csv")
        write_json(
            {
                "status": "PASS",
                "files_hashed": len(hashes),
                "manifest_sha256": file_sha256(audit_dir / "results_hashes.csv"),
                "note": "Manifest excludes itself and its receipt file.",
            },
            audit_dir / "hash_manifest_status.json",
        )
        log(f"E20-E23 PASS: {report_path}")
        return 0
    except Exception as exc:
        write_json(
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            audit_dir / "pipeline_failure.json",
        )
        log(f"E20-E23 FAIL: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
