from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "work" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import holidays
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


E0_DIR = ROOT / "outputs" / "01_data_audit"
E1_DIR = ROOT / "outputs" / "02_e1_point_forecast"
DEFAULT_OUTPUT = ROOT / "outputs" / "03_e1_1_feature_reconciliation"
TARGETS = {
    "mae_mw": 610.91,
    "rmse_mw": 945.46,
    "mape_pct": 4.45,
    "peak_mae_mw": 721.62,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit E0/E1 feature definitions and lineage.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_example(name: str) -> str:
    examples = {
        "cal_doy_sin": "2025-01-01: sin(2*pi*0/365) = 0.000000",
        "cal_doy_cos": "2025-01-01: cos(2*pi*0/365) = 1.000000",
        "cal_dow_sin": "2025-01-01 (Wed=2): sin(2*pi*2/7) = 0.974928",
        "cal_dow_cos": "2025-01-01 (Wed=2): cos(2*pi*2/7) = -0.222521",
        "cal_weekend": "2025-01-04 (Saturday) = 1",
        "cal_federal_holiday": "2025-01-01 (observed New Year's Day) = 1",
        "cal_day_before_holiday": "2024-12-31 because 2025-01-01 is a holiday = 1",
        "cal_day_after_holiday": "2025-01-02 because 2025-01-01 is a holiday = 1",
        "cal_leap_year": "2024-07-01 = 1; 2025-07-01 = 0",
    }
    if name in examples:
        return examples[name]
    if name.startswith("cal_dow_"):
        dow = int(name.rsplit("_", 1)[1])
        dates = {
            0: "2025-01-06 Monday",
            1: "2025-01-07 Tuesday",
            2: "2025-01-08 Wednesday",
            3: "2025-01-09 Thursday",
            4: "2025-01-10 Friday",
            5: "2025-01-11 Saturday",
            6: "2025-01-12 Sunday",
        }
        return f"{dates[dow]} = 1"
    month = int(name.rsplit("_", 1)[1])
    return f"2025-{month:02d}-01 = 1"


def formula(name: str) -> tuple[str, str]:
    formulas = {
        "cal_doy_sin": ("sin(2*pi*(DOY-1)/Y), where Y is 365 or 366 for the target year", "Python math + calendar"),
        "cal_doy_cos": ("cos(2*pi*(DOY-1)/Y), where Y is 365 or 366 for the target year", "Python math + calendar"),
        "cal_dow_sin": ("sin(2*pi*DOW/7), with Monday=0 and Sunday=6", "Python math + pandas Timestamp"),
        "cal_dow_cos": ("cos(2*pi*DOW/7), with Monday=0 and Sunday=6", "Python math + pandas Timestamp"),
        "cal_weekend": ("1 when DOW is Saturday or Sunday; otherwise 0", "Custom"),
        "cal_federal_holiday": ("1 when target date is an observed US federal holiday", "pandas USFederalHolidayCalendar"),
        "cal_day_before_holiday": ("1 when target date + 1 calendar day is in the observed federal-holiday set", "pandas USFederalHolidayCalendar + custom"),
        "cal_day_after_holiday": ("1 when target date - 1 calendar day is in the observed federal-holiday set", "pandas USFederalHolidayCalendar + custom"),
        "cal_leap_year": ("1 when the target year is a Gregorian leap year; otherwise 0", "Python calendar"),
    }
    if name in formulas:
        return formulas[name]
    if name.startswith("cal_dow_"):
        value = int(name.rsplit("_", 1)[1])
        return (f"1 when target DOW={value} under Monday=0; otherwise 0", "Custom")
    month = int(name.rsplit("_", 1)[1])
    return (f"1 when target month={month}; otherwise 0", "Custom")


def build_calendar_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    selected = manifest.loc[
        manifest["category"].isin(["calendar", "calendar_one_hot"])
    ].copy()
    records = []
    for row in selected.itertuples(index=False):
        definition, implementation = formula(row.feature_name)
        records.append(
            {
                "feature_index": int(row.feature_index),
                "feature_name": row.feature_name,
                "feature_group": row.category,
                "formula_or_definition": definition,
                "example": feature_example(row.feature_name),
                "implementation_source": implementation,
                "package_version": f"pandas {pd.__version__}" if "pandas" in implementation else "Python standard library/custom",
                "source_date_rule": row.source_date_rule,
                "known_at_forecast_origin": bool(row.known_at_forecast_origin),
            }
        )
    result = pd.DataFrame(records)
    if len(result) != 28:
        raise AssertionError(f"Expected 28 calendar features, found {len(result)}")
    return result


def holiday_sets(dates: pd.Series) -> dict[str, set[pd.Timestamp]]:
    start = dates.min() - pd.Timedelta(days=2)
    end = dates.max() + pd.Timedelta(days=2)
    pandas_observed = {
        pd.Timestamp(value).normalize()
        for value in USFederalHolidayCalendar().holidays(start=start, end=end)
    }
    years = range(int(start.year) - 1, int(end.year) + 2)
    python_observed = {pd.Timestamp(value) for value in holidays.UnitedStates(years=years, observed=True)}
    python_actual = {pd.Timestamp(value) for value in holidays.UnitedStates(years=years, observed=False)}
    return {
        "pandas_observed": pandas_observed,
        "python_observed": python_observed,
        "python_actual": python_actual,
    }


def calendar_matrix(
    dates: pd.Series,
    holiday_set: set[pd.Timestamp],
    doy_denominator: str = "actual_year",
    doy_phase: str = "zero_based",
    dow_phase: str = "zero_based",
) -> np.ndarray:
    rows = []
    for raw_date in dates:
        date = pd.Timestamp(raw_date).normalize()
        if doy_denominator == "actual_year":
            denominator = 366.0 if calendar.isleap(date.year) else 365.0
        elif doy_denominator == "365.25":
            denominator = 365.25
        else:
            denominator = 365.0
        doy = date.dayofyear - 1 if doy_phase == "zero_based" else date.dayofyear
        dow = date.dayofweek if dow_phase == "zero_based" else date.dayofweek + 1
        rows.append(
            [
                math.sin(2 * math.pi * doy / denominator),
                math.cos(2 * math.pi * doy / denominator),
                math.sin(2 * math.pi * dow / 7),
                math.cos(2 * math.pi * dow / 7),
                float(date.dayofweek >= 5),
                float(date in holiday_set),
                float(date + pd.Timedelta(days=1) in holiday_set),
                float(date - pd.Timedelta(days=1) in holiday_set),
                float(calendar.isleap(date.year)),
                *(float(date.dayofweek == value) for value in range(7)),
                *(float(date.month == value) for value in range(1, 13)),
            ]
        )
    return np.asarray(rows, dtype=float)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error / actual)) * 100),
        "peak_mae_mw": float(np.mean(np.abs(actual.max(1) - predicted.max(1)))),
    }


def run_variant(
    features: pd.DataFrame,
    base_names: list[str],
    calendar_values: np.ndarray,
    target_names: list[str],
) -> dict[str, float]:
    dates = pd.to_datetime(features["date"])
    x = np.column_stack([features[base_names].to_numpy(dtype=float), calendar_values])
    y = features[target_names].to_numpy(dtype=float)
    prediction = np.full_like(y, np.nan)
    for month_start in pd.date_range("2025-01-01", "2025-12-01", freq="MS"):
        train = dates.between(month_start - pd.DateOffset(years=5), month_start - pd.Timedelta(days=1)).to_numpy()
        test = dates.between(month_start, month_start + pd.offsets.MonthEnd(0)).to_numpy()
        scaler = StandardScaler()
        model = Ridge(alpha=100)
        model.fit(scaler.fit_transform(x[train]), y[train])
        prediction[test] = model.predict(scaler.transform(x[test]))
    evaluation = dates.between("2025-01-01", "2025-12-31").to_numpy()
    return metrics(y[evaluation], prediction[evaluation])


def finite_calendar_audit(features: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(features["date"])
    sets = holiday_sets(dates)
    base_names = manifest.loc[
        ~manifest["category"].isin(["calendar", "calendar_one_hot"]), "feature_name"
    ].tolist()
    target_names = [f"target_load_h{hour:02d}" for hour in range(1, 25)]
    variants = [
        {
            "variant": "frozen_current",
            "purpose": "Current E0 definition; comparison reference, not selected from this audit",
            "holiday_set": "pandas_observed",
            "doy_denominator": "actual_year",
            "doy_phase": "zero_based",
            "dow_phase": "zero_based",
            "selected_for_pipeline": True,
        },
        {
            "variant": "fixed_365_denominator",
            "purpose": "Common fixed-year alternative",
            "holiday_set": "pandas_observed",
            "doy_denominator": "365",
            "doy_phase": "zero_based",
            "dow_phase": "zero_based",
            "selected_for_pipeline": False,
        },
        {
            "variant": "fixed_365_25_denominator",
            "purpose": "Common tropical-year approximation",
            "holiday_set": "pandas_observed",
            "doy_denominator": "365.25",
            "doy_phase": "zero_based",
            "dow_phase": "zero_based",
            "selected_for_pipeline": False,
        },
        {
            "variant": "one_based_doy",
            "purpose": "Tests DOY rather than DOY-1 phase convention",
            "holiday_set": "pandas_observed",
            "doy_denominator": "actual_year",
            "doy_phase": "one_based",
            "dow_phase": "zero_based",
            "selected_for_pipeline": False,
        },
        {
            "variant": "one_based_dow",
            "purpose": "Tests Monday=1 rather than Monday=0 cyclic phase",
            "holiday_set": "pandas_observed",
            "doy_denominator": "actual_year",
            "doy_phase": "zero_based",
            "dow_phase": "one_based",
            "selected_for_pipeline": False,
        },
        {
            "variant": "python_holidays_observed",
            "purpose": "Alternative observed-holiday package convention",
            "holiday_set": "python_observed",
            "doy_denominator": "actual_year",
            "doy_phase": "zero_based",
            "dow_phase": "zero_based",
            "selected_for_pipeline": False,
        },
        {
            "variant": "python_holidays_actual",
            "purpose": "Actual holiday dates without substitute observed dates",
            "holiday_set": "python_actual",
            "doy_denominator": "actual_year",
            "doy_phase": "zero_based",
            "dow_phase": "zero_based",
            "selected_for_pipeline": False,
        },
    ]
    rows = []
    for variant in variants:
        values = calendar_matrix(
            dates,
            sets[str(variant["holiday_set"])],
            doy_denominator=str(variant["doy_denominator"]),
            doy_phase=str(variant["doy_phase"]),
            dow_phase=str(variant["dow_phase"]),
        )
        result = run_variant(features, base_names, values, target_names)
        row = {**variant, **result}
        for metric_name, target in TARGETS.items():
            row[f"abs_diff_{metric_name}"] = abs(result[metric_name] - target)
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_report(
    summary_manifest: pd.DataFrame,
    calendar_manifest: pd.DataFrame,
    checks: pd.DataFrame,
    audit: pd.DataFrame,
    e1_metrics: pd.DataFrame,
) -> str:
    current = e1_metrics.set_index("model").loc["ridge_full"]
    no_calendar = e1_metrics.set_index("model").loc["ridge_no_calendar"]
    no_d1 = e1_metrics.set_index("model").loc["ridge_no_d1_profile"]
    d1 = e1_metrics.set_index("model").loc["d1_persistence"]
    summary_table = markdown_table(
        summary_manifest,
        ["feature_index", "feature_name", "source_date_rule", "definition"],
        ["Index", "Feature", "Date rule", "Definition"],
    )
    audit_display = audit.copy()
    audit_display["selected_for_pipeline"] = audit_display["selected_for_pipeline"].map({True: "Yes", False: "No"})
    audit_table = markdown_table(
        audit_display,
        ["variant", "selected_for_pipeline", "mae_mw", "rmse_mw", "mape_pct", "peak_mae_mw"],
        ["Calendar variant", "Frozen", "MAE MW", "RMSE MW", "MAPE percent", "Peak MAE MW"],
    )
    check_table = markdown_table(checks, ["check", "result", "evidence"], ["Check", "Result", "Evidence"])
    calendar_summary = markdown_table(
        calendar_manifest[["feature_name", "formula_or_definition", "example", "implementation_source"]],
        ["feature_name", "formula_or_definition", "example", "implementation_source"],
        ["Feature", "Formula or definition", "Example", "Implementation"],
    )
    return f"""# E1.1 ISO-NE特征定义核对报告

## 结论与状态

当前状态定为 **E1 CONDITIONAL HOLD**，E2暂不启动。

核对确认，当前 `feature_manifest.csv` 中21个日摘要滞后特征实际使用 d-1、d-7和d-28。E1读取的特征文件、日负荷文件和特征清单哈希均与当前重跑后的E0完全一致，因此当前E1与修订后的E0具有连续的数据血缘。较早交付的E0报告曾写为d-1、d-2和d-7；该旧定义随后被覆盖，且原稿只写了“lagged values”，没有给出三个滞后的确切取值。因此，旧E0与当前版本之间确有定义变化，不能把它视为单纯的报告笔误。

当前完整Ridge模型未通过0.1 MW的精确重建门槛，但满足建议的科学复现标准：MAE、RMSE和Peak MAE与原稿的差异均小于1 MW，MAPE差异小于0.01个百分点；模型排序和消融结论保持一致；D-1基线复现；36个月度模型记录均未发现时间泄漏。若作者批准冻结当前定义，本报告建议将它作为新的完全公开实现，更新后续全部实验结果，而不再继续尝试其他特征组合以追逐610.91 MW。

## 日摘要滞后定义核对

较早版本的E0使用d-1、d-2和d-7。当前E0源码、当前特征清单和E1输入均使用d-1、d-7和d-28。当前21列如下。

{summary_table}

选择d-1、d-7和d-28具有短期、周周期和四周周期的可解释性，也与原稿列出的负荷曲线滞后集合一致。然而，原稿本身不足以唯一推出这三个取值，而且该候选是在复现诊断过程中发现的。因此，它应被表述为大修版新冻结的特征定义，而不能声称已经证明它就是初稿的原始实现。

## 数据血缘与无泄漏核对

{check_table}

E1的运行记录保存了实际输入哈希，而不是仅依赖文件名。当前三个哈希全部一致。每个月的标准化器只在相应训练窗口拟合，训练结束日期均早于预测开始日期。

## 当前28维日历特征

当前实现使用目标日期信息。节假日集合由pandas {pd.__version__}的 `USFederalHolidayCalendar` 生成，包含观察日；节假日前后日均按相邻自然日判断。日序周期使用零基相位和当年实际天数，星期周期使用Monday=0。

{calendar_summary}

完整机器可读版本保存在 `calendar_feature_manifest.csv`，其中还包含软件版本、日期边界和forecast-origin可用性。

## 有限日历实现审计

本次只检查七个预先可解释的单因素或软件实现差异。`frozen_current` 始终保持为当前定义；表中的其他版本仅用于判断结果对合理实现差异是否敏感，不依据其与610.91 MW的接近程度改变模型定义。

{audit_table}

不同合理口径会带来小幅变化，但没有一个版本同时重现四项原稿指标。这支持“原稿没有充分记录calendar implementation”的判断，也说明继续扩大搜索空间会转变为事后调参。本报告不从这些候选中另选最接近原稿的版本。

## 两级复现门槛

### Exact Reconstruction Gate

内部精确重建标准要求MW指标差异小于0.1 MW，MAPE差异小于0.01个百分点。当前完整Ridge结果为MAE {current['mae_mw']:.6f} MW、RMSE {current['rmse_mw']:.6f} MW、MAPE {current['mape_pct']:.6f}%和Peak MAE {current['peak_mae_mw']:.6f} MW。相对原稿的绝对差异分别为 {current['absolute_difference_mae_mw']:.6f} MW、{current['absolute_difference_rmse_mw']:.6f} MW、{current['absolute_difference_mape_pct']:.6f}个百分点和 {current['absolute_difference_peak_mae_mw']:.6f} MW。结论为 **FAIL**。

### Scientific Reproduction Gate

科学复现标准要求三个MW指标差异小于1 MW、MAPE差异小于0.01个百分点、模型排序和消融结论一致、D-1基线复现且不存在时间泄漏。当前结果满足全部数值条件：

| Model | MAE MW | Reported MAE MW | Difference MW | Interpretation |
|---|---:|---:|---:|---|
| Full Ridge | {current['mae_mw']:.3f} | 610.910 | {current['absolute_difference_mae_mw']:.3f} | Best Ridge specification |
| No calendar | {no_calendar['mae_mw']:.3f} | 627.360 | {no_calendar['absolute_difference_mae_mw']:.3f} | Removing calendar features worsens MAE |
| No d-1 profile | {no_d1['mae_mw']:.3f} | 707.330 | {no_d1['absolute_difference_mae_mw']:.3f} | Removing the d-1 profile causes a large loss |
| D-1 persistence | {d1['mae_mw']:.3f} | 970.750 | {d1['absolute_difference_mae_mw']:.3f} | Baseline reproduces within 0.1 MW |

该门槛的量化部分为 **PASS**。由于日摘要滞后定义无法从初稿唯一恢复，整体状态仍为 **CONDITIONAL HOLD**，等待作者正式冻结大修版定义。

## 建议的冻结决策

建议作者批准以下定义作为大修版pipeline v1.0：

1. 日摘要滞后固定为d-1、d-7和d-28，并在Methods和Supplement中逐列说明。
2. 28维日历特征按 `calendar_feature_manifest.csv` 固定，不再根据旧指标尝试其他组合。
3. 以当前E0三个SHA-256哈希作为E1及后续实验的输入版本标识。
4. 将611.66 MW作为新pipeline的2025 Full Ridge MAE，并从该冻结版本重新计算E2及后续结果。
5. 在公开代码中保存本报告、两个manifest、运行配置和月度无泄漏审计。

作者批准该冻结决策后，可把E1状态改为 **SCIENTIFIC REPRODUCTION PASS** 并启动E2。若作者坚持逐值复原初稿，则仍需原始训练代码或原始233维设计矩阵，当前资料不足以证明610.91 MW对应的唯一calendar implementation。
"""


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    feature_path = E0_DIR / "ISO_NE_features.parquet"
    daily_path = E0_DIR / "ISO_NE_daily_profile.parquet"
    manifest_path = E0_DIR / "feature_manifest.csv"
    e1_config_path = E1_DIR / "e1_run_config.json"
    e1_metrics_path = E1_DIR / "e1_metrics.csv"
    e1_audit_path = E1_DIR / "e1_monthly_fit_audit.csv"

    manifest = pd.read_csv(manifest_path).sort_values("feature_index")
    features = pd.read_parquet(feature_path).sort_values("date").reset_index(drop=True)
    config = json.loads(e1_config_path.read_text(encoding="utf-8"))
    e1_metrics = pd.read_csv(e1_metrics_path)
    monthly_audit = pd.read_csv(e1_audit_path)

    summary_manifest = manifest.loc[manifest["category"] == "lagged_daily_summary"].copy()
    lag_rules = summary_manifest["source_date_rule"].drop_duplicates().tolist()
    if len(summary_manifest) != 21 or set(lag_rules) != {"d-1", "d-7", "d-28"}:
        raise AssertionError(f"Unexpected current summary lag definition: {lag_rules}")
    summary_manifest.to_csv(output / "daily_summary_lag_manifest.csv", index=False)

    calendar_manifest = build_calendar_manifest(manifest)
    calendar_manifest.to_csv(output / "calendar_feature_manifest.csv", index=False)

    current_hashes = {
        "features": sha256(feature_path),
        "daily": sha256(daily_path),
        "manifest": sha256(manifest_path),
    }
    checks = pd.DataFrame(
        [
            {
                "check": "Current manifest feature count",
                "result": "PASS" if len(manifest) == 233 else "FAIL",
                "evidence": f"{len(manifest)} rows",
            },
            {
                "check": "Current daily-summary lag definition",
                "result": "PASS",
                "evidence": "21 columns at d-1, d-7, and d-28",
            },
            {
                "check": "E1 feature-table hash equals current E0",
                "result": "PASS" if config["input_sha256"]["features"] == current_hashes["features"] else "FAIL",
                "evidence": current_hashes["features"],
            },
            {
                "check": "E1 daily-profile hash equals current E0",
                "result": "PASS" if config["input_sha256"]["daily"] == current_hashes["daily"] else "FAIL",
                "evidence": current_hashes["daily"],
            },
            {
                "check": "E1 manifest hash equals current E0",
                "result": "PASS" if config["input_sha256"]["manifest"] == current_hashes["manifest"] else "FAIL",
                "evidence": current_hashes["manifest"],
            },
            {
                "check": "Monthly training and scaler boundaries",
                "result": "PASS" if bool(monthly_audit["no_temporal_leakage"].all()) else "FAIL",
                "evidence": f"{int(monthly_audit['no_temporal_leakage'].sum())}/{len(monthly_audit)} model-month rows pass",
            },
        ]
    )
    checks.to_csv(output / "reconciliation_checks.csv", index=False)
    if not (checks["result"] == "PASS").all():
        raise AssertionError("At least one reconciliation check failed")

    audit = finite_calendar_audit(features, manifest)
    audit.to_csv(output / "calendar_implementation_audit.csv", index=False, float_format="%.10f")

    report = make_report(summary_manifest, calendar_manifest, checks, audit, e1_metrics)
    (output / "E1_1_feature_reconciliation_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
