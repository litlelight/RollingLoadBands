from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / "work" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd

from experiments.revision_common import (
    StageTimer,
    audit_oos_and_calibration,
    band_metrics_for_period,
    build_band_metrics,
    build_conformal_store,
    decision_analysis,
    file_sha256,
    freeze_check,
    generate_oos_ridge,
    hardware_snapshot,
    load_feature_inputs,
    load_yaml,
    log,
    multiyear_point_metrics,
    output_hash_manifest,
    paired_block_bootstrap,
    paired_point_model_bootstrap,
    seasonal_naive_results,
    software_snapshot,
    write_csv,
    write_json,
    write_parquet,
    lambda_stability,
)
from experiments.revision_models import run_tcn_experiment, run_xgboost_experiment


def _read_config(config_path: str | Path) -> dict[str, Any]:
    return load_yaml(Path(config_path))


def _stage_row(name: str, started: float, status: str = "PASS", detail: str = "") -> dict[str, Any]:
    return {
        "stage": name,
        "status": status,
        "runtime_seconds": time.perf_counter() - started,
        "detail": detail,
    }


def run_cpu_branch(config_path: str, result_root_text: str) -> dict[str, Any]:
    """Run all methods that depend on the audited ISO-NE forecast store, plus NYISO."""
    config = _read_config(config_path)
    result_root = Path(result_root_text)
    e0_dir = ROOT / "outputs" / "01_data_audit"
    a_dir = result_root / "A_oos_and_audit"
    b_dir = result_root / "B_conformal"
    c_dir = result_root / "C_decision_and_bootstrap"
    d_dir = result_root / "D_baselines"
    e_dir = result_root / "E_sensitivity"
    f_dir = result_root / "F_nyiso_external"
    timings: list[dict[str, Any]] = []

    features, daily, manifest = load_feature_inputs(e0_dir, "ISO_NE")
    store = pd.read_parquet(a_dir / "oos_prediction_store.parquet")

    started = time.perf_counter()
    log("E4-E7: building four conformal variants from immutable OOS residuals")
    bands = build_conformal_store(store, config["conformal"])
    band_metrics = build_band_metrics(store, bands, daily)
    write_parquet(bands, b_dir / "conformal_prediction_store.parquet")
    write_csv(band_metrics, b_dir / "conformal_metrics_by_period.csv")
    timings.append(_stage_row("E4-E7 conformal variants", started))

    started = time.perf_counter()
    log("E8-E10/E14-E15: decision cube and paired moving-block bootstrap")
    decision_cube, decision_daily, decision_tuning = decision_analysis(
        store, bands, daily, config["decision"]
    )
    bootstrap = paired_block_bootstrap(decision_daily, config["bootstrap"], int(config["random_seed"]))
    write_csv(decision_cube, c_dir / "decision_cube_2025.csv")
    write_parquet(decision_daily, c_dir / "decision_daily_2025.parquet")
    write_csv(decision_tuning, c_dir / "decision_tuning_2024.csv")
    write_csv(bootstrap, c_dir / "paired_moving_block_bootstrap.csv")
    timings.append(_stage_row("E8-E10 and E14-E15 decision analysis", started))

    started = time.perf_counter()
    log("E11: D-1 and D-7 seasonal-naive baselines")
    naive_predictions, naive_metrics = seasonal_naive_results(daily)
    write_parquet(naive_predictions, d_dir / "seasonal_naive_predictions.parquet")
    write_csv(naive_metrics, d_dir / "seasonal_naive_metrics.csv")
    timings.append(_stage_row("E11 seasonal-naive baselines", started))

    started = time.perf_counter()
    log("E16-E17: lambda stability and multi-year rolling diagnostics")
    stability = lambda_stability(
        store,
        bands,
        daily,
        config["decision"]["cost_ratios"],
        config["decision"]["lambda_grid"],
    )
    point_multi = multiyear_point_metrics(store, "Ridge")
    combined_point = pd.concat([point_multi, naive_metrics], ignore_index=True)
    write_csv(stability, e_dir / "lambda_stability_2020_2024.csv")
    write_csv(combined_point, e_dir / "multiyear_point_metrics.csv")
    write_csv(band_metrics, e_dir / "multiyear_conformal_metrics.csv")
    timings.append(_stage_row("E16-E17 multi-year diagnostics", started))

    started = time.perf_counter()
    log("E18: NYISO external validation with frozen ISO-NE method settings")
    ny_features, ny_daily, ny_manifest = load_feature_inputs(e0_dir, "NYISO")
    ny_config = config["nyiso"]
    ny_store, ny_fit_audit = generate_oos_ridge(
        ny_features,
        ny_manifest,
        ny_config["oos_start"],
        ny_config["oos_end"],
        float(config["ridge"]["alpha"]),
        int(config["ridge"]["training_years"]),
        "NYISO",
        int(config["hardware"]["cpu_workers"]),
    )
    history_methods = {
        "standardized_max_90": 90,
        "one_sided_max_90": 90,
        "disjoint_standardized_max_90": 180,
        "bonferroni_365": 365,
    }
    ny_audit_summary, ny_violations, ny_gate = audit_oos_and_calibration(
        ny_store,
        ny_fit_audit,
        ny_config["validation_start"],
        ny_config["evaluation_end"],
        history_methods,
    )
    write_parquet(ny_store, f_dir / "nyiso_oos_prediction_store.parquet")
    write_csv(ny_fit_audit, f_dir / "nyiso_monthly_fit_audit.csv")
    write_csv(ny_audit_summary, f_dir / "nyiso_calibration_audit_summary.csv")
    write_csv(ny_violations, f_dir / "nyiso_calibration_audit_violations.csv")
    if not ny_gate:
        raise AssertionError("NYISO OOS/calibration gate failed; see NYISO audit files")
    ny_bands = build_conformal_store(ny_store, config["conformal"])
    ny_metrics_rows = []
    for method in sorted(ny_bands["method"].unique()):
        for label, start_date, end_date in (
            ("2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
            ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
        ):
            record = band_metrics_for_period(ny_store, ny_bands, ny_daily, method, start_date, end_date)
            record["period"] = label
            ny_metrics_rows.append(record)
    ny_band_metrics = pd.DataFrame(ny_metrics_rows)
    ny_cube, ny_daily_decision, ny_tuning = decision_analysis(
        ny_store, ny_bands, ny_daily, config["decision"]
    )
    ny_bootstrap = paired_block_bootstrap(
        ny_daily_decision, config["bootstrap"], int(config["random_seed"]) + 100_000
    )
    ny_naive_predictions, ny_naive_metrics_all = seasonal_naive_results(ny_daily)
    ny_naive_metrics = ny_naive_metrics_all.loc[ny_naive_metrics_all["period"].isin(["2024", "2025"])]
    ny_point = multiyear_point_metrics(ny_store, "Ridge")
    ny_point = ny_point.loc[ny_point["period"].isin(["2024", "2025"])]
    write_parquet(ny_bands, f_dir / "nyiso_conformal_prediction_store.parquet")
    write_csv(ny_band_metrics, f_dir / "nyiso_conformal_metrics.csv")
    write_csv(ny_cube, f_dir / "nyiso_decision_cube_2025.csv")
    write_parquet(ny_daily_decision, f_dir / "nyiso_decision_daily_2025.parquet")
    write_csv(ny_tuning, f_dir / "nyiso_decision_tuning_2024.csv")
    write_csv(ny_bootstrap, f_dir / "nyiso_paired_moving_block_bootstrap.csv")
    write_csv(pd.concat([ny_point, ny_naive_metrics], ignore_index=True), f_dir / "nyiso_point_metrics.csv")
    write_parquet(
        ny_naive_predictions.loc[pd.to_datetime(ny_naive_predictions["target_date"]).between("2024-01-01", "2025-12-31")],
        f_dir / "nyiso_seasonal_naive_predictions_2024_2025.parquet",
    )
    write_json(
        {
            "status": "PASS",
            "method_transfer": "All feature logic, Ridge alpha, five-year rolling window, conformal settings, grids, and bootstrap procedures were transferred unchanged from ISO-NE.",
            "known_source_gap": "Raw NYISO coverage has an early gap from 2021-04-01 through 2021-06-02; the 2023-2025 OOS and 2024-2025 calibration/evaluation dates are complete.",
        },
        f_dir / "nyiso_external_validation_status.json",
    )
    timings.append(_stage_row("E18 NYISO external validation", started))

    write_csv(pd.DataFrame(timings), result_root / "G_reproducibility" / "cpu_branch_runtime.csv")
    return {
        "status": "PASS",
        "timings": timings,
        "iso_band_rows": len(bands),
        "iso_decision_rows": len(decision_cube),
        "nyiso_oos_rows": len(ny_store),
    }


def run_gpu_branch(config_path: str, result_root_text: str) -> dict[str, Any]:
    """Run the same-feature nonlinear comparator and the fully identified five-seed TCN."""
    config = _read_config(config_path)
    result_root = Path(result_root_text)
    e0_dir = ROOT / "outputs" / "01_data_audit"
    d_dir = result_root / "D_baselines"
    features, daily, manifest = load_feature_inputs(e0_dir, "ISO_NE")
    timings: list[dict[str, Any]] = []

    started = time.perf_counter()
    log("E12: same-feature XGBoost development and frozen evaluation")
    xgb_summary = run_xgboost_experiment(
        features,
        manifest,
        config["xgboost"],
        d_dir / "xgboost",
        int(config["random_seed"]),
        int(config["hardware"]["cpu_workers"]),
    )
    timings.append(_stage_row("E12 same-feature XGBoost", started))

    started = time.perf_counter()
    log("E13: fixed TCN architecture across five named seeds")
    tcn_summary = run_tcn_experiment(daily, features, manifest, config["tcn"], d_dir / "tcn")
    timings.append(_stage_row("E13 five-seed TCN", started))
    write_csv(pd.DataFrame(timings), result_root / "G_reproducibility" / "gpu_branch_runtime.csv")
    return {"status": "PASS", "timings": timings, "xgboost": xgb_summary, "tcn": tcn_summary}


def _format_float(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str], labels: list[str] | None = None) -> str:
    subset = frame.loc[:, columns].copy()
    if labels:
        subset.columns = labels
    headers = [str(column) for column in subset.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in subset.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(_format_float(value, 3))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def _create_figures(result_root: Path, report_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    point = pd.read_csv(result_root / "E_sensitivity" / "multiyear_point_metrics.csv")
    xgb = pd.read_csv(result_root / "D_baselines" / "xgboost" / "xgboost_selected_metrics.csv")
    tcn = pd.read_csv(result_root / "D_baselines" / "tcn" / "tcn_seed_metrics.csv")
    values = []
    for model in ("Ridge", "D-1", "D-7"):
        row = point.loc[(point["model"] == model) & (point["period"].astype(str) == "2025")].iloc[0]
        values.append((model, float(row["mae_mw"])))
    values.append(("XGBoost", float(xgb.loc[xgb["period"].astype(str) == "2025", "mae_mw"].iloc[0])))
    values.append(("TCN mean", float(tcn["mae_mw"].mean())))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar([item[0] for item in values], [item[1] for item in values], color=["#355C7D", "#C06C84", "#F67280", "#6C5B7B", "#2A9D8F"])
    ax.set_ylabel("2025 MAE (MW)")
    ax.set_title("Point forecast comparison on the held-out 2025 period")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "point_model_mae_2025.png", dpi=180)
    plt.close(fig)

    bands = pd.read_csv(result_root / "B_conformal" / "conformal_metrics_by_period.csv")
    subset = bands.loc[bands["period"].astype(str) == "2025"].sort_values("method")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(subset["method"], subset["daily_upper_coverage_pct"], color="#457B9D")
    ax.axhline(90, color="#E63946", linestyle="--", linewidth=1.5, label="Nominal 90%")
    ax.set_ylabel("Daily simultaneous upper coverage (%)")
    ax.set_title("2025 conformal upper-band coverage")
    ax.tick_params(axis="x", rotation=22)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "conformal_upper_coverage_2025.png", dpi=180)
    plt.close(fig)

    cube = pd.read_csv(result_root / "C_decision_and_bootstrap" / "decision_cube_2025.csv")
    selected = cube.loc[(cube["peak_threshold"] == "historical_q95") & (cube["cost_ratio"] == 10)].copy()
    selected = selected.loc[selected["method"].isin(["fixed_point", "tuned_point", "two_sided_conformal", "one_sided_conformal"])]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(selected["method"], selected["cost"], color="#F4A261")
    ax.set_ylabel("Cost = FP + 10 × FN")
    ax.set_title("2025 peak-alert cost at the historical 95th-percentile threshold")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "decision_cost_q95_c10_2025.png", dpi=180)
    plt.close(fig)


def _build_final_report(config: dict[str, Any], result_root: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    point = pd.read_csv(result_root / "E_sensitivity" / "multiyear_point_metrics.csv")
    bands = pd.read_csv(result_root / "B_conformal" / "conformal_metrics_by_period.csv")
    cube = pd.read_csv(result_root / "C_decision_and_bootstrap" / "decision_cube_2025.csv")
    boot = pd.read_csv(result_root / "C_decision_and_bootstrap" / "paired_moving_block_bootstrap.csv")
    stability = pd.read_csv(result_root / "E_sensitivity" / "lambda_stability_2020_2024.csv")
    xgb_candidates = pd.read_csv(result_root / "D_baselines" / "xgboost" / "xgboost_development_candidates.csv")
    xgb = pd.read_csv(result_root / "D_baselines" / "xgboost" / "xgboost_selected_metrics.csv")
    tcn = pd.read_csv(result_root / "D_baselines" / "tcn" / "tcn_seed_metrics.csv")
    tcn_architecture = json.loads((result_root / "D_baselines" / "tcn" / "tcn_architecture_and_run.json").read_text(encoding="utf-8"))
    point_bootstrap = pd.read_csv(result_root / "D_baselines" / "point_model_paired_bootstrap.csv")
    ny_point = pd.read_csv(result_root / "F_nyiso_external" / "nyiso_point_metrics.csv")
    ny_bands = pd.read_csv(result_root / "F_nyiso_external" / "nyiso_conformal_metrics.csv")
    freeze = pd.read_csv(result_root / "00_freeze" / "freeze_check.csv")
    gate = json.loads((result_root / "A_oos_and_audit" / "gate_status.json").read_text(encoding="utf-8"))

    p2025 = point.loc[point["period"].astype(str) == "2025"].copy()
    p2025 = p2025.loc[p2025["model"].isin(["Ridge", "D-1", "D-7"])]
    x2025 = xgb.loc[xgb["period"].astype(str) == "2025"].copy()
    x2025["model"] = "XGBoost"
    aggregate = tcn_architecture["aggregate_metrics"]
    tcn_row = pd.DataFrame([
        {"model": "TCN seeds (mean ± SD)", "mae_mw": f"{tcn['mae_mw'].mean():.2f} ± {tcn['mae_mw'].std(ddof=1):.2f}", "rmse_mw": f"{tcn['rmse_mw'].mean():.2f} ± {tcn['rmse_mw'].std(ddof=1):.2f}", "mape_pct": f"{tcn['mape_pct'].mean():.3f} ± {tcn['mape_pct'].std(ddof=1):.3f}", "peak_mae_mw": f"{tcn['peak_mae_mw'].mean():.2f} ± {tcn['peak_mae_mw'].std(ddof=1):.2f}"},
        {"model": "TCN five-seed ensemble", **aggregate},
    ])
    model_table = pd.concat(
        [p2025[["model", "mae_mw", "rmse_mw", "mape_pct", "peak_mae_mw"]], x2025[["model", "mae_mw", "rmse_mw", "mape_pct", "peak_mae_mw"]], tcn_row],
        ignore_index=True,
    )

    b2025 = bands.loc[bands["period"].astype(str) == "2025"].copy()
    decision = cube.loc[(cube["peak_threshold"] == "historical_q95") & (cube["cost_ratio"] == 10)].copy()
    decision = decision.loc[decision["method"].isin(["fixed_point", "tuned_point", "two_sided_conformal", "one_sided_conformal", "equal_alert_point_two_sided", "equal_alert_point_one_sided"])]
    primary_boot = boot.loc[
        (boot["peak_threshold"] == "historical_q95")
        & (boot["cost_ratio"] == 10)
        & (boot["block_length_days"] == int(config["bootstrap"]["primary_block_days"]))
        & (boot["metric"] == "cost")
        & (boot["method_b"] == "tuned_point")
    ].copy()
    point_boot_primary = point_bootstrap.loc[
        (point_bootstrap["block_length_days"] == int(config["bootstrap"]["primary_block_days"]))
        & (point_bootstrap["metric"] == "mae_mw")
        & (point_bootstrap["model_a"].isin(["D-1", "D-7", "XGBoost balanced", "TCN five-seed ensemble"]))
    ].copy()

    ny_2025_point = ny_point.loc[ny_point["period"].astype(str) == "2025"]
    ny_2025_bands = ny_bands.loc[ny_bands["period"].astype(str) == "2025"]

    lines = [
        f"# {config['pipeline_version']} 全套大修实验报告",
        "",
        "## 执行结论",
        "",
        f"整套 E2–E19 已执行完成。ISO-NE 的 OOS 残差与时间边界硬门禁状态为 **{gate['status']}**；冻结输入的 {len(freeze)} 项哈希/结构检查全部通过。2025 始终作为最终评估期，2024 仅用于决策参数选择；测试期没有用于超参数或阈值选择。",
        "",
        "本报告中的 coverage 是滚动、依赖数据下的经验覆盖率，不表述为交换性条件下的严格有限样本保证。Equal-alert comparator 明确标记为 ex-post 诊断，不作为可部署规则。",
        "",
        "## 1. 2025 点预测结果",
        "",
        _markdown_table(model_table, ["model", "mae_mw", "rmse_mw", "mape_pct", "peak_mae_mw"], ["模型", "MAE (MW)", "RMSE (MW)", "MAPE (%)", "峰值 MAE (MW)"]),
        "",
        "![2025 点预测比较](figures/point_model_mae_2025.png)",
        "",
        f"XGBoost 的候选模型只依据 2020–2023 rolling-origin MAE 选择；入选配置为 **{xgb_candidates.sort_values('mae_mw').iloc[0]['candidate']}**。TCN 使用固定架构和 seeds = 11, 22, 33, 42, 55；所有无法追溯的旧版 ‘original TCN prediction path’ 均不再使用。",
        "",
        "下表给出 2025 日级误差的 paired moving-block bootstrap（7-day block，5,000 次）。差值定义为 comparator − Ridge，正值表示 Ridge 的误差更低。每个 TCN seed 的结果及 block=3/14/28 天敏感性均保存在完整 CSV。",
        "",
        _markdown_table(point_boot_primary, ["model_a", "model_b", "observed_difference", "ci_lower", "ci_upper", "probability_difference_gt_zero"], ["Comparator", "Reference", "MAE 差值 (MW)", "95% CI 下限", "95% CI 上限", "P(差值>0)"]),
        "",
        "## 2. 2025 Conformal 区间/上界",
        "",
        _markdown_table(
            b2025,
            ["method", "daily_upper_coverage_pct", "hourly_upper_coverage_pct", "peak_hour_upper_coverage_pct", "mean_upper_margin_mw", "daily_simultaneous_coverage_pct", "mean_width_mw"],
            ["方法", "逐日联合上界覆盖率 (%)", "逐小时上界覆盖率 (%)", "峰值小时上界覆盖率 (%)", "平均上边际 (MW)", "双侧逐日联合覆盖率 (%)", "平均宽度 (MW)"],
        ),
        "",
        "![2025 上界覆盖率](figures/conformal_upper_coverage_2025.png)",
        "",
        "方法口径：M1 为 90-day standardized-max；M2 为 365-day corrected Bonferroni；M3 为 one-sided standardized-max；M4 用 d-180:d-91 估计尺度、d-90:d-1 校准 score。所有 q、尺度和 score 均只引用目标日前已经产生的 OOS 残差。",
        "",
        "## 3. 2025 峰值预警主比较（历史 95% 阈值，漏报代价比 10）",
        "",
        _markdown_table(
            decision,
            ["method", "setting_type", "setting", "TP", "FP", "FN", "TN", "alerts", "precision", "recall", "false_alarm_rate", "cost", "is_ex_post"],
            ["规则", "调参类型", "参数", "TP", "FP", "FN", "TN", "预警数", "Precision", "Recall", "FAR", "Cost", "Ex-post"],
        ),
        "",
        "![2025 决策成本](figures/decision_cost_q95_c10_2025.png)",
        "",
        "主 paired moving-block bootstrap（7-day block，5,000 次）对 Cost 差值的结果如下；差值定义为 conformal − tuned point，负值表示 conformal 成本更低。",
        "",
        _markdown_table(primary_boot, ["method_a", "method_b", "observed_difference", "ci_lower", "ci_upper", "probability_difference_lt_zero"], ["方法 A", "方法 B", "观察差值", "95% CI 下限", "95% CI 上限", "P(差值<0)"]),
        "",
        "完整的 5 个阈值口径 × 5 个成本比 × 所有主要规则位于 decision cube；block=3/7/14/28 天的配对不确定性位于 bootstrap 表。",
        "",
        "## 4. 多年份与 λ 稳定性",
        "",
        "2020–2024 只解释为 retrospective rolling-origin diagnostics，不包装成新的独立测试集；2025 是主测试期，2026H1 是追加的时间外诊断。",
        "",
        _markdown_table(stability, ["year", "band", "cost_ratio", "selected_lambda", "cost", "recall", "false_alarm_rate"], ["年份", "上界类型", "成本比", "λ", "Cost", "Recall", "FAR"]),
        "",
        "## 5. NYISO 外部验证",
        "",
        "NYISO 沿用 ISO-NE 冻结后的 233 维特征逻辑、Ridge α=100、5 年滚动训练、calibration windows、决策网格和 bootstrap 方法。NYISO 的 2024 年只用于调参，2025 年用于 external evaluation。原始归档在 2021-04-01 至 2021-06-02 有早期缺口，但 2023–2025 OOS 以及 2024–2025 校准/评估日期连续，并已通过独立门禁。",
        "",
        _markdown_table(ny_2025_point, ["model", "days", "mae_mw", "rmse_mw", "mape_pct", "peak_mae_mw"], ["模型", "天数", "MAE (MW)", "RMSE (MW)", "MAPE (%)", "峰值 MAE (MW)"]),
        "",
        _markdown_table(ny_2025_bands, ["method", "daily_upper_coverage_pct", "hourly_upper_coverage_pct", "peak_hour_upper_coverage_pct", "mean_upper_margin_mw"], ["方法", "逐日联合上界覆盖率 (%)", "逐小时上界覆盖率 (%)", "峰值小时覆盖率 (%)", "平均上边际 (MW)"]),
        "",
        "## 6. 可复现性与文件索引",
        "",
        "- `00_freeze/`: 冻结配置、输入哈希和 E1.1 接受记录。",
        "- `A_oos_and_audit/`: E2 OOS forecast store、逐月模型审计、E3 门禁。",
        "- `B_conformal/`: E4–E7 四种 conformal 方法的逐日完整输出和年度指标。",
        "- `C_decision_and_bootstrap/`: E8–E10、E14–E15 的调参、逐日决策、decision cube 和 paired bootstrap。",
        "- `D_baselines/`: E11 D-1/D-7、E12 XGBoost、E13 五 seed TCN。",
        "- `E_sensitivity/`: E16 λ 稳定性和 E17 多年份结果。",
        "- `F_nyiso_external/`: E18 NYISO 全套外部验证。",
        "- `G_reproducibility/`: E19 硬件、软件、运行时间、状态和全部文件 SHA-256。",
        "",
        "具体论文取舍应以 bootstrap CI、跨年份稳定性和 NYISO 方向一致性共同判断，不能只凭单个 2025 点估计下结论。",
        "",
    ]
    report_path = report_dir / "FINAL_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete staged major-revision experiment suite.")
    parser.add_argument("--config", default="configs/revision_v1.yaml")
    parser.add_argument("--pipeline-version", default=None)
    parser.add_argument("--no-parallel", action="store_true", help="Run CPU and GPU branches sequentially.")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = _read_config(config_path)
    version = args.pipeline_version or str(config["pipeline_version"])
    if version != str(config["pipeline_version"]):
        raise ValueError("--pipeline-version must match the frozen config")

    result_root = ROOT / "results" / version
    report_dir = ROOT / "reports" / version
    for name in (
        "00_freeze",
        "A_oos_and_audit",
        "B_conformal",
        "C_decision_and_bootstrap",
        "D_baselines",
        "E_sensitivity",
        "F_nyiso_external",
        "G_reproducibility",
    ):
        (result_root / name).mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    g_dir = result_root / "G_reproducibility"
    e0_dir = ROOT / "outputs" / "01_data_audit"
    all_started = time.perf_counter()
    main_timings: list[dict[str, Any]] = []
    write_json(hardware_snapshot(), g_dir / "hardware.json")
    write_json(software_snapshot(), g_dir / "software_versions.json")
    shutil.copy2(config_path, result_root / "00_freeze" / "revision_v1.yaml")

    try:
        started = time.perf_counter()
        log("Freeze check: validating E1.1 feature definition, hashes, and 611.66-MW pipeline identity")
        frozen = freeze_check(config, e0_dir)
        write_csv(frozen, result_root / "00_freeze" / "freeze_check.csv")
        write_json(
            {
                "status": "ACCEPTED_AND_FROZEN",
                "pipeline_version": version,
                "decision": "The reconciled d-1/d-7/d-28, 28-calendar-feature pipeline is the sole basis for E2-E19.",
                "reference_e1_mae_mw": 611.655997,
            },
            result_root / "00_freeze" / "freeze_decision.json",
        )
        main_timings.append(_stage_row("freeze check", started))

        started = time.perf_counter()
        log("E2: generating ISO-NE monthly rolling-origin OOS forecast store (2019-2026H1)")
        features, daily, manifest = load_feature_inputs(e0_dir, "ISO_NE")
        store, fit_audit = generate_oos_ridge(
            features,
            manifest,
            config["ridge"]["oos_start"],
            config["ridge"]["oos_end"],
            float(config["ridge"]["alpha"]),
            int(config["ridge"]["training_years"]),
            "ISO_NE",
            int(config["hardware"]["cpu_workers"]),
        )
        a_dir = result_root / "A_oos_and_audit"
        write_parquet(store, a_dir / "oos_prediction_store.parquet")
        write_csv(fit_audit, a_dir / "monthly_fit_audit.csv")
        main_timings.append(_stage_row("E2 ISO-NE OOS forecast store", started))

        started = time.perf_counter()
        log("E3: enforcing leakage and calibration-source hard gate")
        history_methods = {
            "standardized_max_90": 90,
            "one_sided_max_90": 90,
            "disjoint_standardized_max_90": 180,
            "bonferroni_365": 365,
        }
        audit_summary, violations, gate = audit_oos_and_calibration(
            store,
            fit_audit,
            "2020-01-01",
            config["ridge"]["oos_end"],
            history_methods,
        )
        write_csv(audit_summary, a_dir / "calibration_audit_summary.csv")
        write_csv(violations, a_dir / "calibration_audit_violations.csv")
        write_json(
            {
                "status": "PASS" if gate else "FAIL",
                "rule": "If E2/E3 fails, conformal and decision experiments must not run.",
                "oos_rows": len(store),
                "oos_start": store["target_date"].min(),
                "oos_end": store["target_date"].max(),
                "violations": len(violations),
            },
            a_dir / "gate_status.json",
        )
        main_timings.append(_stage_row("E3 leakage/calibration hard gate", started, "PASS" if gate else "FAIL"))
        if not gate:
            raise AssertionError("E2/E3 FAIL: downstream experiments were intentionally stopped")
        log("E3 PASS: all downstream branches are authorized")

        branch_started = time.perf_counter()
        if args.no_parallel:
            cpu_result = run_cpu_branch(str(config_path), str(result_root))
            gpu_result = run_gpu_branch(str(config_path), str(result_root))
        else:
            log("Launching CPU analytics/NYISO branch alongside the CUDA XGBoost/TCN branch")
            with ProcessPoolExecutor(max_workers=2) as executor:
                cpu_future = executor.submit(run_cpu_branch, str(config_path), str(result_root))
                gpu_future = executor.submit(run_gpu_branch, str(config_path), str(result_root))
                cpu_result = cpu_future.result()
                gpu_result = gpu_future.result()
        main_timings.append(_stage_row("parallel downstream branches", branch_started))
        write_json(cpu_result, g_dir / "cpu_branch_summary.json")
        write_json(gpu_result, g_dir / "gpu_branch_summary.json")

        started = time.perf_counter()
        log("E13 audit: paired moving-block comparison of every named point-model path")
        ridge_store = pd.read_parquet(result_root / "A_oos_and_audit" / "oos_prediction_store.parquet")
        naive = pd.read_parquet(result_root / "D_baselines" / "seasonal_naive_predictions.parquet")
        comparators: dict[str, pd.DataFrame] = {
            "D-1": naive.loc[naive["model"] == "D-1"],
            "D-7": naive.loc[naive["model"] == "D-7"],
            "XGBoost balanced": pd.read_parquet(result_root / "D_baselines" / "xgboost" / "xgboost_selected_predictions.parquet"),
            "TCN five-seed ensemble": pd.read_parquet(result_root / "D_baselines" / "tcn" / "tcn_five_seed_mean_predictions.parquet"),
        }
        for seed in config["tcn"]["seeds"]:
            comparators[f"TCN seed {seed}"] = pd.read_parquet(
                result_root / "D_baselines" / "tcn" / f"tcn_seed_{seed}_predictions.parquet"
            )
        point_bootstrap = paired_point_model_bootstrap(
            ridge_store, comparators, config["bootstrap"], int(config["random_seed"])
        )
        write_csv(point_bootstrap, result_root / "D_baselines" / "point_model_paired_bootstrap.csv")
        main_timings.append(_stage_row("E13 point-model paired bootstrap", started))

        started = time.perf_counter()
        log("E19: creating figures, final report, runtime inventory, and cryptographic result manifest")
        _create_figures(result_root, report_dir)
        report_path = _build_final_report(config, result_root, report_dir)
        main_timings.append(_stage_row("E19 report generation", started))
        main_timings.append(
            {
                "stage": "complete pipeline wall time",
                "status": "PASS",
                "runtime_seconds": time.perf_counter() - all_started,
                "detail": "Wall time; parallel branch durations are reported separately.",
            }
        )
        write_csv(pd.DataFrame(main_timings), g_dir / "main_runtime.csv")
        status = {
            "status": "PASS",
            "pipeline_version": version,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "wall_time_seconds": time.perf_counter() - all_started,
            "final_report": str(report_path),
        }
        write_json(status, g_dir / "pipeline_status.json")
        hashes = output_hash_manifest(result_root)
        write_csv(hashes, g_dir / "results_hashes.csv")
        write_json(
            {
                "status": "PASS",
                "files_hashed": len(hashes),
                "manifest_sha256": file_sha256(g_dir / "results_hashes.csv"),
                "note": "The manifest excludes itself; report figures live in reports/ and are not included in this result-tree manifest.",
            },
            g_dir / "hash_manifest_status.json",
        )
        log(f"PIPELINE PASS: {report_path}")
        return 0
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "pipeline_version": version,
            "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "wall_time_seconds": time.perf_counter() - all_started,
        }
        write_json(failure, g_dir / "pipeline_failure.json")
        write_csv(pd.DataFrame(main_timings), g_dir / "main_runtime_partial.csv")
        log(f"PIPELINE FAIL: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
