"""Run second-round reviewer experiments without rewriting frozen v1.x outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.review_round2 import (
    ROOT,
    UPPER_COLUMNS,
    band_quality,
    build_peak_bands,
    build_rolling_bands,
    decision_grid,
    load_inputs,
    nyiso_target_audit,
    seasonal_diagnostics,
    select_retrospective_2024,
)
from experiments.revision_common import file_sha256, load_yaml, log, write_csv, write_json, write_parquet


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
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


def _gate_new_bands(name: str, old: pd.DataFrame, bands: pd.DataFrame) -> None:
    date_mask = pd.to_datetime(old.target_date).between("2024-01-01", "2025-12-31")
    for old_method, new_method in (("standardized_max_90", "two_sided_w90"),
                                   ("one_sided_max_90", "one_sided_w90")):
        left = old.loc[date_mask & (old.method == old_method)].sort_values("target_date")
        right = bands.loc[bands.method == new_method].sort_values("target_date")
        if len(left) != 731 or len(right) != 731 or not np.array_equal(
            pd.to_datetime(left.target_date).to_numpy(), pd.to_datetime(right.target_date).to_numpy()
        ):
            raise AssertionError(f"{name} {new_method}: reference date gate")
        difference = np.max(np.abs(left[UPPER_COLUMNS].to_numpy(float) - right[UPPER_COLUMNS].to_numpy(float)))
        if difference > 1e-6:
            raise AssertionError(f"{name} {new_method}: reference upper mismatch {difference}")
    eta_zero = bands.loc[bands.method == "aci_eta0"].sort_values("target_date")
    primary = bands.loc[bands.method == "two_sided_w90"].sort_values("target_date")
    difference = np.max(np.abs(eta_zero[UPPER_COLUMNS].to_numpy(float)
                               - primary[UPPER_COLUMNS].to_numpy(float)))
    if difference > 1e-6:
        raise AssertionError(f"{name}: zero-step ACI not identical to fixed-alpha rolling band")


def _gate_new_decisions(name: str, summary: pd.DataFrame) -> dict[str, float]:
    primary = summary.loc[(summary.threshold_spec == "historical_q95")
                          & np.isclose(summary.cost_ratio, 10)]
    expected = {"ISO_NE": {"tuned_point": 24, "two_sided_w90": 17, "one_sided_w90": 16},
                "NYISO": {"tuned_point": 11, "two_sided_w90": 22, "one_sided_w90": 20}}[name]
    costs = {}
    for method, value in expected.items():
        selected = primary.loc[primary.method == method]
        if len(selected) != 1 or not np.isclose(float(selected.test_cost.iloc[0]), value):
            raise AssertionError(f"{name}: main decision gate failed for {method}")
        costs[method] = float(selected.test_cost.iloc[0])
    if not (primary.evaluation_events == 17).all():
        raise AssertionError(f"{name}: frozen Q95 event count changed")
    return costs


def _matrix_pattern(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary.loc[summary.method.isin(["tuned_point", "two_sided_w90", "one_sided_w90"])]
    pivot = selected.pivot(index=["dataset", "threshold_spec", "cost_ratio"],
                           columns="method", values="test_cost").reset_index()
    rows = []
    for dataset, group in pivot.groupby("dataset"):
        for method in ("two_sided_w90", "one_sided_w90"):
            difference = group[method] - group.tuned_point
            rows.append({"dataset": dataset, "method": method, "operating_settings": len(group),
                         "lower_cost": int((difference < 0).sum()),
                         "same_cost": int((difference == 0).sum()),
                         "higher_cost": int((difference > 0).sum())})
    return pd.DataFrame(rows)


def _gate_daily_decisions(summary: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Independently reconcile every published decision row to its 2025 daily trace."""
    keys = ["dataset", "threshold_spec", "cost_ratio", "method"]
    trace = daily.assign(
        TP=daily.event & daily.alert,
        FP=~daily.event & daily.alert,
        FN=daily.event & ~daily.alert,
        TN=~daily.event & ~daily.alert,
    ).groupby(keys, as_index=False).agg(
        trace_days=("target_date", "size"),
        trace_TP=("TP", "sum"), trace_FP=("FP", "sum"),
        trace_FN=("FN", "sum"), trace_TN=("TN", "sum"),
        trace_alerts=("alert", "sum"), trace_cost=("daily_loss", "sum"),
    )
    checked = summary.merge(trace, on=keys, how="outer", validate="one_to_one", indicator=True)
    if len(checked) != len(summary) or not (checked._merge == "both").all():
        raise AssertionError("Daily trace and decision-matrix row keys differ")
    if not (checked.trace_days == 365).all():
        raise AssertionError("A decision setting lacks 365 evaluation days")
    if daily.duplicated(keys + ["target_date"]).any():
        raise AssertionError("Duplicate daily decision key")
    for metric in ("TP", "FP", "FN", "TN", "alerts", "cost"):
        if not np.allclose(checked[f"test_{metric}"], checked[f"trace_{metric}"], rtol=0, atol=1e-9):
            raise AssertionError(f"Daily trace does not reproduce test_{metric}")
    return checked[keys + ["trace_days", "trace_TP", "trace_FP", "trace_FN", "trace_TN",
                           "trace_alerts", "trace_cost"]]


def _make_report(root: Path, report_root: Path, config: dict, quality: pd.DataFrame,
                 decision: pd.DataFrame, selection: pd.DataFrame, seasonal: pd.DataFrame,
                 episodes: pd.DataFrame, resampling: pd.DataFrame, nyiso: pd.DataFrame, tie: pd.DataFrame,
                 pattern: pd.DataFrame, dependence: pd.DataFrame, ranks: pd.DataFrame,
                 elapsed: float) -> Path:
    q = quality.loc[(quality.period == 2025)
                    & quality.method.isin(["two_sided_w90", "one_sided_w90", "aci_eta0",
                                            "direct_peak_w90", "direct_peak_w365"])].copy()
    d = decision.loc[(decision.threshold_spec == "historical_q95")
                     & np.isclose(decision.cost_ratio, 10)
                     & decision.method.isin(["tuned_point", "two_sided_w30", "two_sided_w90",
                                             "two_sided_w365", "one_sided_w30", "one_sided_w90",
                                             "one_sided_w365", "direct_peak_w90", "direct_peak_w365"])].copy()
    d = d.sort_values(["dataset", "family", "window_days"])
    chosen_windows = selection.loc[selection.family.isin(["two_sided", "one_sided"])].copy()
    aci_q = quality.loc[(quality.period == 2025) & (quality.family == "aci_projected")].copy()
    aci_d = decision.loc[(decision.threshold_spec == "historical_q95")
                         & np.isclose(decision.cost_ratio, 10)
                         & (decision.family == "aci_projected")].copy()
    aci_comparison = aci_q.merge(
        aci_d[["dataset", "method", "selected_parameter", "validation_cost", "test_cost"]],
        on=["dataset", "method"], validate="one_to_one",
    )
    selected_aci = dict(zip(selection.loc[selection.family == "aci_projected", "dataset"],
                            selection.loc[selection.family == "aci_projected", "method"]))
    aci_comparison["chosen_from_2024_coverage"] = aci_comparison.apply(
        lambda row: row.method == selected_aci[row.dataset], axis=1
    )
    event_counts = decision.loc[(decision.method == "tuned_point")
                                & np.isclose(decision.cost_ratio, 10),
                                ["dataset", "threshold_spec", "evaluation_events"]].copy()
    seasonal_focus = seasonal.loc[((seasonal.dataset == "ISO_NE") & (seasonal.month == 3))
                                  | ((seasonal.dataset == "NYISO") & (seasonal.month == 6))]
    resampling_focus = resampling.loc[(resampling.block_days == 7)
                                      & resampling.metric.isin(["paired_annual_cost_difference",
                                                                "daily_joint_coverage"])].copy()
    dependence_focus = dependence.loc[(dependence.method == "two_sided_w90")
                                       & dependence.series.isin(["joint_failure", "paired_daily_loss"])
                                       & dependence.lag_days.isin([1, 7, 14, 28])]
    episode_focus = episodes.loc[episodes.episode_paired_cost != 0]
    aci_findings = []
    for dataset, chosen_method in selected_aci.items():
        adaptive = aci_comparison.loc[(aci_comparison.dataset == dataset)
                                      & (aci_comparison.method == chosen_method)].iloc[0]
        rolling = q.loc[(q.dataset == dataset) & (q.method == "two_sided_w90")].iloc[0]
        rolling_cost = d.loc[(d.dataset == dataset) & (d.method == "two_sided_w90")].test_cost.iloc[0]
        aci_findings.append(
            f"{dataset}: 2024-selected {chosen_method} has 2025 joint coverage "
            f"{adaptive.joint_coverage:.2%}, width {adaptive.mean_width_mw:.0f} MW and cost "
            f"{adaptive.test_cost:.0f}; ordinary rolling 90-day has {rolling.joint_coverage:.2%}, "
            f"{rolling.mean_width_mw:.0f} MW and cost {rolling_cost:.0f}."
        )
    lines = [
        "# ROUND2_REVISION_V1 — second-round reviewer experiments", "",
        "These are retrospective analyses of frozen historical out-of-sample forecasts. The 90-day rows retain",
        "the inherited reference setting. Selecting a window from 2024 is a newly specified retrospective",
        "exercise, not an original prespecified protocol or an untouched 2025 confirmation.", "",
        "## Gates and scope", "",
        "Frozen ISO-NE hashes, both original 2025 coverage counts, and primary decision costs passed.",
        f"All {len(decision)} decision-matrix rows were independently reconciled with 365 daily records each.",
        "The new 90-day band paths agree with the prior paths to within 1e-6 MW.",
        "Zero-step projected ACI agrees with the ordinary 90-day two-sided path.",
        f"Runtime: {elapsed:.1f} s. Seed: {config['random_seed']}.", "",
        "## Calibration-window decision sensitivity", "",
        _markdown_table(d, ["dataset", "method", "window_days", "selected_parameter",
                            "validation_cost", "test_TP", "test_FP", "test_FN", "test_cost"]), "",
        "Full 30/60/90/180/270/365-day tables for both systems, both band families, and all 25",
        "threshold/cost settings are in E25/window_decision_matrix.csv. The 2024 joint window/",
        "lambda selection rows are in E25/retrospective_window_selection_2024.csv.", "",
        "### Window-plus-lambda choices using 2024 cost", "",
        _markdown_table(chosen_windows, ["dataset", "method", "window_days", "selected_parameter",
                                         "validation_cost", "test_cost"]), "",
        "These selections were formulated for the revision after the 2025 record had been inspected.", "",
        "## Sequential and direct-daily-maximum baselines", "",
        _markdown_table(q, ["dataset", "method", "period", "days", "joint_coverage",
                            "upper_coverage", "peak_upper_coverage", "mean_width_mw",
                            "interval_score_mw", "mean_peak_margin_mw"]), "",
        "ACI is a projected sequential alpha-update baseline with the same archived Ridge forecasts",
        "and 90-day maximum score. Its step-size grid and 2024-only retrospective selection are saved.",
        "The full ACI step-size comparison at historical Q95 and cost ratio 10 is:", "",
        _markdown_table(aci_comparison, ["dataset", "method", "chosen_from_2024_coverage",
                                         "joint_coverage", "mean_width_mw", "interval_score_mw",
                                         "boundary_days", "selected_parameter", "validation_cost",
                                         "test_cost"]), "",
        *aci_findings, "",
        "Direct-peak calibration uses historical residuals of the daily predicted maximum and does not",
        "produce a 24-position profile band.", "",
        "## Full decision-matrix pattern relative to tuned point", "",
        _markdown_table(pattern, ["dataset", "method", "operating_settings", "lower_cost",
                                  "same_cost", "higher_cost"]), "",
        "These 25 settings are correlated scenarios rather than 25 independent replications.", "",
        "### Peak-event denominators by threshold", "",
        _markdown_table(event_counts, ["dataset", "threshold_spec", "evaluation_events"]), "",
        "## Calendar and resampling diagnostics", "",
        _markdown_table(seasonal_focus, ["dataset", "month", "method", "days", "events",
                                         "joint_covered_days", "upper_covered_days",
                                         "paired_cost_difference"]), "",
        _markdown_table(resampling_focus, ["dataset", "method", "metric", "design",
                                          "block_days", "observed", "resample_p025",
                                          "resample_p975"]), "",
        "Daily error/loss dependence diagnostics used to inspect the block-length choice:", "",
        _markdown_table(dependence_focus, ["dataset", "series", "lag_days", "autocorrelation"]), "",
        "Leave-one-peak-episode-out diagnostic for episodes with nonzero paired cost:", "",
        _markdown_table(episode_focus, ["dataset", "method", "start", "end", "event_days",
                                        "episode_paired_cost", "annual_paired_cost",
                                        "annual_difference_without_episode"]), "",
        "The observed 2025 annual total is a fixed-calendar descriptive quantity. The resampling",
        "ranges describe hypothetical reweighted day sequences under each stated block design,",
        "conditional on the archived forecasts and validation-selected rules.", "",
        "The resampling ranges condition on the stored predictions and selected rules. Annual circular",
        "blocks wrap across December/January; the alternatives do not. None is presented as a",
        "confidence interval for a fixed realized year or an assured future-year population.", "",
        "## NYISO target-definition audit", "",
        _markdown_table(nyiso, ["test_days", "peak_events", "irregular_days", "irregular_peak_days",
                                "interpolated_days", "interpolated_peak_days", "dst_adjusted_days",
                                "dst_adjusted_peak_days"]), "",
        "Irregular timestamps are distinct local timestamp/time-zone pairs appearing off the five-minute",
        "grid. Equal-observation hourly averaging is the evaluated target, not a duration-weighted mean.", "",
        "## Tie-breaking audit", "",
        f"{int(tie.tie_rule_changes_choice.sum())} of {len(tie)} point-rule operating settings select a",
        "different delta if the manuscript's smaller-parameter rule replaces the implemented",
        "absolute-parameter-first rule. Primary results still use the released implementation.", "",
        "All implemented second-round quantile ranks are finite and at most their calibration counts.",
        "Projected ACI state boundaries and per-method rank ranges are in E29/quantile_rank_audit.csv.", "",
        "## Machine-readable outputs", "",
        "E24/input_hashes.csv, daily_trace_gate.csv; E25/{dataset}/window_and_aci_bands.parquet,",
        "E25/window_decision_matrix.csv, all_decision_daily.parquet, primary_decision_daily.parquet;",
        "E26/quality_metrics.csv and E26/{dataset}/direct_peak_daily.parquet;",
        "E28/monthly_diagnostics.csv, resampling_sensitivity.csv, event_episode_loo.csv,",
        "daily_error_dependence.csv; E29/nyiso_target_audit.csv, tie_break_audit.csv,",
        "quantile_rank_audit.csv.", "",
    ]
    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / "ROUND2_EXPERIMENT_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/review_round2_v1.yaml")
    parser.add_argument("--nyiso-raw-dir", default="data/raw_nyiso",
                        help="Directory containing the twelve official 2025 NYISO monthly ZIP archives")
    args = parser.parse_args()
    nyiso_raw_dir = Path(args.nyiso_raw_dir).expanduser()
    if not nyiso_raw_dir.is_absolute():
        nyiso_raw_dir = ROOT / nyiso_raw_dir
    config = load_yaml(ROOT / args.config)
    result_root = ROOT / "results" / config["pipeline_version"]
    report_root = ROOT / "reports" / config["pipeline_version"]
    started = time.perf_counter()
    log("E24: checking frozen inputs and previously reported primary results")
    sources, hashes = load_inputs(config)
    (result_root / "E24").mkdir(parents=True, exist_ok=True)
    write_csv(hashes, result_root / "E24" / "input_hashes.csv")
    quality_parts = []
    decision_parts = []
    daily_parts = []
    tie_parts = []
    monthly_parts = []
    seasonal_daily_parts = []
    episode_parts = []
    bootstrap_parts = []
    dependence_parts = []
    peak_parts = []
    rank_parts = []
    gates = []
    for name, source in sources.items():
        log(f"E25/E26/E27: {name} rolling windows, ACI, direct peak, and decision tuning")
        store, daily = source["store"], source["daily"]
        bands = build_rolling_bands(store, config)
        peak_bands = build_peak_bands(store, config)
        _gate_new_bands(name, source["old_bands"], bands)
        if not ((bands.quantile_rank >= 1) & (bands.quantile_rank <= bands.window_days)).all():
            raise AssertionError(f"{name}: out-of-range quantile rank")
        for method, frame in bands.groupby("method"):
            rank_parts.append({"dataset": name, "method": method, "family": frame.family.iloc[0],
                               "window_days": int(frame.window_days.iloc[0]),
                               "min_rank": int(frame.quantile_rank.min()),
                               "max_rank": int(frame.quantile_rank.max()),
                               "min_alpha_state": float(frame.alpha_state.min()) if frame.alpha_state.notna().any() else np.nan,
                               "max_alpha_state": float(frame.alpha_state.max()) if frame.alpha_state.notna().any() else np.nan,
                               "boundary_days": int(frame.state_at_boundary.sum()),
                               "out_of_range_ranks": 0})
        band_daily, quality = band_quality(store, bands, peak_bands, float(config["alpha"]))
        decisions, decision_daily, tie = decision_grid(name, store, daily, bands, peak_bands, config)
        primary_daily_system = decision_daily.loc[
            (decision_daily.threshold_spec == "historical_q95")
            & np.isclose(decision_daily.cost_ratio, 10)
        ].copy()
        costs = _gate_new_decisions(name, decisions)
        gates.append({"dataset": name, "status": "PASS", **costs})
        dataset_root = result_root / "E25" / name
        write_parquet(bands, dataset_root / "window_and_aci_bands.parquet")
        write_parquet(band_daily, dataset_root / "band_daily_quality.parquet")
        write_parquet(peak_bands, result_root / "E26" / name / "direct_peak_daily.parquet")
        log(f"E28: {name} monthly diagnostics and calendar-sensitive resampling")
        monthly, seasonal_daily, episodes, boot, dependence = seasonal_diagnostics(
            name, band_daily.loc[band_daily.target_date.between("2025-01-01", "2025-12-31")],
            primary_daily_system, config,
        )
        quality_parts.append(quality)
        decision_parts.append(decisions)
        daily_parts.append(decision_daily)
        tie_parts.append(tie)
        monthly_parts.append(monthly)
        seasonal_daily_parts.append(seasonal_daily)
        episode_parts.append(episodes)
        bootstrap_parts.append(boot)
        dependence_parts.append(dependence)
        peak_parts.append(peak_bands)
    quality = pd.concat(quality_parts, ignore_index=True)
    decision = pd.concat(decision_parts, ignore_index=True)
    decision_daily = pd.concat(daily_parts, ignore_index=True)
    trace_gate = _gate_daily_decisions(decision, decision_daily)
    primary_decision_daily = decision_daily.loc[
        (decision_daily.threshold_spec == "historical_q95")
        & np.isclose(decision_daily.cost_ratio, 10)
    ].copy()
    tie = pd.concat(tie_parts, ignore_index=True)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    seasonal_daily = pd.concat(seasonal_daily_parts, ignore_index=True)
    episodes = pd.concat(episode_parts, ignore_index=True)
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)
    dependence = pd.concat(dependence_parts, ignore_index=True)
    peak = pd.concat(peak_parts, ignore_index=True)
    ranks = pd.DataFrame(rank_parts)
    selection = select_retrospective_2024(decision, quality)
    pattern = _matrix_pattern(decision)
    log("E29: auditing NYISO raw test-year timestamps and tie behavior")
    nyiso_daily, nyiso_summary = nyiso_target_audit(
        primary_decision_daily.loc[primary_decision_daily.dataset == "NYISO"], nyiso_raw_dir
    )
    write_csv(pd.DataFrame(gates), result_root / "E24" / "baseline_gates.csv")
    write_csv(trace_gate, result_root / "E24" / "daily_trace_gate.csv")
    write_csv(quality, result_root / "E26" / "quality_metrics.csv")
    write_csv(decision, result_root / "E25" / "window_decision_matrix.csv")
    write_parquet(decision_daily, result_root / "E25" / "all_decision_daily.parquet")
    write_parquet(primary_decision_daily, result_root / "E25" / "primary_decision_daily.parquet")
    write_csv(selection, result_root / "E25" / "retrospective_window_selection_2024.csv")
    write_csv(pattern, result_root / "E25" / "primary_window_matrix_pattern.csv")
    write_csv(monthly, result_root / "E28" / "monthly_diagnostics.csv")
    write_parquet(seasonal_daily, result_root / "E28" / "seasonal_daily.parquet")
    write_csv(episodes, result_root / "E28" / "event_episode_loo.csv")
    write_csv(bootstrap, result_root / "E28" / "resampling_sensitivity.csv")
    write_csv(dependence, result_root / "E28" / "daily_error_dependence.csv")
    write_csv(nyiso_daily, result_root / "E29" / "nyiso_target_daily.csv")
    write_csv(nyiso_summary, result_root / "E29" / "nyiso_target_audit.csv")
    write_csv(tie, result_root / "E29" / "tie_break_audit.csv")
    write_csv(ranks, result_root / "E29" / "quantile_rank_audit.csv")
    elapsed = time.perf_counter() - started
    path = _make_report(result_root, report_root, config, quality, decision, selection, monthly,
                        episodes, bootstrap, nyiso_summary, tie, pattern, dependence, ranks, elapsed)
    outputs = sorted(p for p in result_root.rglob("*")
                     if p.is_file() and p != result_root / "RUN_STATUS.json")
    write_json({"status": "PASS", "version": config["pipeline_version"],
                "runtime_seconds": elapsed, "report": path.relative_to(ROOT).as_posix(),
                "output_hashes": {str(p.relative_to(result_root)).replace("\\", "/"): file_sha256(p)
                                  for p in outputs}}, result_root / "RUN_STATUS.json")
    log(f"PASS: second-round experiments completed in {elapsed:.1f} s; report {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
