"""Retrospective NYISO aggregation sensitivity without changing frozen E0--E29 inputs.

The alternative hourly target holds each reported system-load observation forward
until the next timestamp (at most 30 minutes) and averages over elapsed time.
It is a sensitivity definition, not a claim about the operator's physical
measurement convention. On an exact five-minute grid it equals the original
equal-observation hourly mean.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = PROJECT_ROOT / "work" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import pyarrow  # noqa: F401 -- initialize Parquet backend before the project modules
import numpy as np
import pandas as pd

from experiments.e0_preprocess import (
    NYISO_END,
    NYISO_START,
    NYISO_ZONES,
    build_model_features,
    canonical_nyiso_archives,
    interpolate_short_gaps,
    normalize_nyiso_days,
    parse_nyiso,
)
from experiments.revision_common import (
    LOAD_COLUMNS,
    PRED_COLUMNS,
    RESIDUAL_COLUMNS,
    TRUE_COLUMNS,
    band_metrics_for_period,
    build_conformal_store,
    decision_analysis,
    file_sha256,
    generate_oos_ridge,
    historical_peak_threshold,
    load_yaml,
    write_csv,
    write_parquet,
)


def raw_system_observations(downloads: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeat E0's source validation and short-zone-gap repair, before hourly means."""
    observations: list[pd.DataFrame] = []
    imputation_rows: list[dict[str, object]] = []
    archive_rows: list[dict[str, object]] = []
    for archive in canonical_nyiso_archives(downloads):
        archive_rows.append({
            "archive": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": file_sha256(archive),
        })
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise ValueError(f"Corrupt source ZIP: {archive}")
            names = sorted(name for name in zipped.namelist() if name.lower().endswith(".csv"))
            for name in names:
                with zipped.open(name) as stream:
                    frame = pd.read_csv(stream)
                if list(frame.columns) != ["Time Stamp", "Time Zone", "Name", "PTID", "Load"]:
                    raise AssertionError(f"Unexpected source schema: {archive.name}/{name}")
                frame = frame.rename(columns={
                    "Time Stamp": "timestamp_local", "Time Zone": "timezone",
                    "Name": "zone", "Load": "load_mw",
                })
                frame["timestamp_local"] = pd.to_datetime(
                    frame["timestamp_local"], format="%m/%d/%Y %H:%M:%S", errors="raise"
                )
                if set(frame["timezone"].astype(str)) - {"EDT", "EST"}:
                    raise AssertionError(f"Unexpected time-zone label in {archive.name}/{name}")
                frame["load_mw"] = pd.to_numeric(frame["load_mw"], errors="coerce")
                frames.append(frame[["timestamp_local", "timezone", "zone", "load_mw"]])
        month = pd.concat(frames, ignore_index=True)
        if set(month["zone"]) != set(NYISO_ZONES):
            raise AssertionError(f"NYISO zone set changed in {archive.name}")
        month["timestamp_utc"] = (
            month["timestamp_local"]
            + pd.to_timedelta(month["timezone"].map({"EDT": 4, "EST": 5}), unit="h")
        ).dt.tz_localize("UTC")
        if month.duplicated(["timestamp_utc", "zone"]).any():
            raise AssertionError(f"Duplicate UTC/zone observation in {archive.name}")
        counts = month.groupby("timestamp_utc")["zone"].nunique()
        if not (counts == len(NYISO_ZONES)).all():
            raise AssertionError(f"Incomplete zone group in {archive.name}")
        meta = (
            month[["timestamp_utc", "timestamp_local", "timezone"]]
            .drop_duplicates("timestamp_utc").sort_values("timestamp_utc")
        )
        pivot = month.pivot(index="timestamp_utc", columns="zone", values="load_mw")
        pivot = pivot.reindex(columns=list(NYISO_ZONES)).sort_index()
        pivot, month_imputations, _ = interpolate_short_gaps(pivot, meta, archive.name)
        imputation_rows.extend(month_imputations)
        system = pivot.sum(axis=1, min_count=len(NYISO_ZONES)).rename("system_load_mw").reset_index()
        system = system.merge(meta, on="timestamp_utc", validate="one_to_one")
        observations.append(system)
    result = pd.concat(observations, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)
    if result["timestamp_utc"].duplicated().any() or result["system_load_mw"].isna().any():
        raise AssertionError("Duplicate or incomplete NYISO system-level timestamps")
    return result, pd.DataFrame(imputation_rows), pd.DataFrame(archive_rows)


def forward_duration_hourly(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Integrate each observed value over [timestamp, next timestamp) in UTC."""
    seconds = observations["timestamp_utc"].dt.as_unit("ns").astype("int64").to_numpy() // 1_000_000_000
    values = observations["system_load_mw"].to_numpy(dtype=float)
    if not np.all(np.diff(seconds) > 0):
        raise AssertionError("Source timestamps are not strictly increasing")
    base_hour = int(seconds[0] // 3600)
    last_hour = int(seconds[-1] // 3600)
    n_hours = last_hour - base_hour + 1
    exposure = np.zeros(n_hours, dtype=float)
    load_seconds = np.zeros(n_hours, dtype=float)

    start = seconds[:-1]
    stop = seconds[1:]
    gap = stop - start
    good = gap <= 30 * 60
    first_hour = start[good] // 3600
    first_end = np.minimum(stop[good], (first_hour + 1) * 3600)
    first_duration = (first_end - start[good]).astype(float)
    idx = (first_hour - base_hour).astype(int)
    np.add.at(exposure, idx, first_duration)
    np.add.at(load_seconds, idx, first_duration * values[:-1][good])

    cross = stop[good] > first_end
    second_duration = (stop[good][cross] - first_end[cross]).astype(float)
    idx2 = idx[cross] + 1
    np.add.at(exposure, idx2, second_duration)
    np.add.at(load_seconds, idx2, second_duration * values[:-1][good][cross])

    # Do not bridge the seven-day official gap or any other gap above 30 min.
    # At a gap boundary, hold at most five minutes to complete an adjacent hour.
    gap_rows: list[dict[str, object]] = []
    for position in np.flatnonzero(~good):
        prev_time = int(start[position])
        next_time = int(stop[position])
        prev_hour = prev_time // 3600
        next_hour = next_time // 3600
        tail = (prev_hour + 1) * 3600 - prev_time
        head = next_time - next_hour * 3600
        if 0 < tail <= 300:
            loc = int(prev_hour - base_hour)
            exposure[loc] += tail
            load_seconds[loc] += tail * values[position]
        if 0 < head <= 300:
            loc = int(next_hour - base_hour)
            exposure[loc] += head
            load_seconds[loc] += head * values[position + 1]
        gap_rows.append({
            "previous_utc": observations.at[position, "timestamp_utc"],
            "next_utc": observations.at[position + 1, "timestamp_utc"],
            "gap_minutes": gap[position] / 60.0,
            "previous_hour_edge_fill_seconds": tail if 0 < tail <= 300 else 0,
            "next_hour_edge_fill_seconds": head if 0 < head <= 300 else 0,
        })

    first_head = int(seconds[0] - base_hour * 3600)
    last_tail = int((last_hour + 1) * 3600 - seconds[-1])
    if first_head > 300 or last_tail > 300:
        raise AssertionError("Dataset edge is more than five minutes from an hour boundary")
    exposure[0] += first_head
    load_seconds[0] += first_head * values[0]
    exposure[-1] += last_tail
    load_seconds[-1] += last_tail * values[-1]

    complete = np.isclose(exposure, 3600.0, rtol=0, atol=1e-6)
    partial = (exposure > 0) & ~complete
    if partial.any():
        sample = (np.flatnonzero(partial)[:10] + base_hour).tolist()
        raise AssertionError(f"Incomplete time-weighted UTC hours: {sample}")
    indices = np.flatnonzero(complete)
    utc_hour = pd.to_datetime((indices + base_hour) * 3600, unit="s", utc=True)
    local_hour = utc_hour.tz_convert("America/New_York")
    local_naive = local_hour.tz_localize(None)
    hourly = pd.DataFrame({
        "timestamp_utc_hour": utc_hour,
        "date": local_naive.normalize(),
        "hour": local_naive.hour,
        "timezone": [stamp.tzname() for stamp in local_hour],
        "source_archive": local_naive.strftime("%Y%m01pal_csv.zip"),
        "system_load_mw": load_seconds[indices] / 3600.0,
        "covered_seconds": exposure[indices],
    })
    return hourly, pd.DataFrame(gap_rows)


def compare_profiles(reference: pd.DataFrame, alternative: pd.DataFrame) -> pd.DataFrame:
    joined = reference[["date", *LOAD_COLUMNS]].merge(
        alternative[["date", *LOAD_COLUMNS]], on="date", suffixes=("_equal", "_duration"),
        validate="one_to_one",
    )
    for column in LOAD_COLUMNS:
        joined[f"delta_{column}_mw"] = joined[f"{column}_duration"] - joined[f"{column}_equal"]
    equal_cols = [f"{column}_equal" for column in LOAD_COLUMNS]
    duration_cols = [f"{column}_duration" for column in LOAD_COLUMNS]
    joined["equal_peak_mw"] = joined[equal_cols].max(axis=1)
    joined["duration_peak_mw"] = joined[duration_cols].max(axis=1)
    joined["peak_delta_mw"] = joined.duration_peak_mw - joined.equal_peak_mw
    joined["max_hour_absolute_delta_mw"] = joined[[f"delta_{column}_mw" for column in LOAD_COLUMNS]].abs().max(axis=1)
    return joined


def event_audit(comparison: pd.DataFrame, original_threshold: float, weighted_threshold: float) -> pd.DataFrame:
    result = comparison[["date", "equal_peak_mw", "duration_peak_mw", "peak_delta_mw"]].copy()
    result["original_threshold_mw"] = original_threshold
    result["duration_threshold_mw"] = weighted_threshold
    result["original_event"] = result.equal_peak_mw > original_threshold
    result["duration_fixed_threshold_event"] = result.duration_peak_mw > original_threshold
    result["duration_reestimated_threshold_event"] = result.duration_peak_mw > weighted_threshold
    return result


def primary_decisions(cube: pd.DataFrame, scenario: str) -> pd.DataFrame:
    methods = ("tuned_point", "two_sided_conformal", "one_sided_conformal")
    selected = cube.loc[
        cube.peak_threshold.eq("historical_q95")
        & np.isclose(pd.to_numeric(cube.cost_ratio), 10.0)
        & cube.method.isin(methods),
        ["method", "fixed_threshold_mw", "setting_type", "setting", "TP", "FP", "FN", "alerts", "cost"],
    ].copy()
    selected.insert(0, "scenario", scenario)
    if len(selected) != 3:
        raise AssertionError(f"Missing primary decision rows for {scenario}")
    return selected


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(downloads: Path, output: Path, root: Path, cpu_threads: int = 12) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frozen_dir = root / "outputs" / "01_data_audit"
    original = pd.read_parquet(frozen_dir / "NYISO_daily_profile.parquet")
    regenerated, _, _, _ = parse_nyiso(downloads)
    if not original.date.equals(regenerated.date):
        raise AssertionError("Current raw ZIP dates do not match the frozen daily profile")
    discrepancy = np.nanmax(np.abs(
        original[LOAD_COLUMNS].to_numpy(float) - regenerated[LOAD_COLUMNS].to_numpy(float)
    ))
    if discrepancy > 1e-7:
        raise AssertionError(f"Current operator ZIPs differ from frozen E0 input: {discrepancy:.9g} MW")

    observations, imputation_log, archive_hashes = raw_system_observations(downloads)
    hourly, long_gaps = forward_duration_hourly(observations)
    hourly_for_normalizer = hourly[["date", "hour", "timezone", "source_archive", "system_load_mw"]]
    available_weighted, dst = normalize_nyiso_days(hourly_for_normalizer, imputation_log)
    calendar = pd.DataFrame({"date": pd.date_range(NYISO_START, NYISO_END, freq="D")})
    weighted = calendar.merge(available_weighted, on="date", how="left", validate="one_to_one")
    weighted["available"] = weighted["available"].fillna(False).astype(bool)
    for column in ("source_file", "gap_reason", "dst_type"):
        weighted[column] = weighted[column].fillna("")
    for column in ("has_imputation", "dst_adjusted"):
        weighted[column] = weighted[column].fillna(False).astype(bool)
    weighted["imputed_zone_cell_count"] = weighted["imputed_zone_cell_count"].fillna(0).astype(int)
    if not original.available.equals(weighted.available):
        altered = original.loc[original.available.ne(weighted.available), "date"].astype(str).tolist()
        raise AssertionError(f"Duration weighting changed available dates: {altered[:10]}")
    if int(weighted.available.sum()) != 2915 or len(dst) != 16:
        raise AssertionError("Daily coverage or DST normalization changed")

    # Independent check: on exactly regular 12-sample hours, hold-forward
    # integration should equal the original equal-observation mean.
    equal_hourly = observations.assign(
        timestamp_utc_hour=observations.timestamp_utc.dt.floor("h")
    ).groupby("timestamp_utc_hour", as_index=False).agg(
        equal_hourly_mw=("system_load_mw", "mean"), samples=("system_load_mw", "size")
    )
    hour_check = hourly.merge(equal_hourly, on="timestamp_utc_hour", validate="one_to_one")
    hour_check["delta_mw"] = hour_check.system_load_mw - hour_check.equal_hourly_mw
    regular = observations.assign(
        timestamp_utc_hour=observations.timestamp_utc.dt.floor("h"),
        on_grid=observations.timestamp_utc.dt.minute.mod(5).eq(0)
        & observations.timestamp_utc.dt.second.eq(0),
    ).groupby("timestamp_utc_hour", as_index=False).agg(
        all_on_grid=("on_grid", "all"), samples=("system_load_mw", "size")
    )
    hour_check = hour_check.merge(regular[["timestamp_utc_hour", "all_on_grid"]], on="timestamp_utc_hour")
    regular_check = hour_check.loc[hour_check.all_on_grid & hour_check.samples.eq(12)]
    regular_error = float(regular_check.delta_mw.abs().max())
    if regular_error > 1e-7:
        raise AssertionError(f"Regular-hour equivalence failed: {regular_error:.9g} MW")

    comparison = compare_profiles(original, weighted)
    original_threshold = historical_peak_threshold(original, pd.Timestamp("2024-01-01"), 95.0)
    weighted_threshold = historical_peak_threshold(weighted, pd.Timestamp("2024-01-01"), 95.0)
    events = event_audit(comparison, original_threshold, weighted_threshold)

    features, invalid = build_model_features(weighted, NYISO_START, NYISO_END, "NYISO")
    frozen_features = pd.read_parquet(frozen_dir / "NYISO_features.parquet")
    if len(features) != len(frozen_features) or not features.date.equals(frozen_features.date):
        raise AssertionError("Time weighting changed feature target dates")
    manifest = pd.read_csv(frozen_dir / "feature_manifest.csv").sort_values("feature_index")
    base_config = load_yaml(root / "configs" / "revision_v1.yaml")
    store, fit_audit = generate_oos_ridge(
        features, manifest,
        base_config["nyiso"]["oos_start"], base_config["nyiso"]["oos_end"],
        float(base_config["ridge"]["alpha"]), int(base_config["ridge"]["training_years"]),
        "NYISO", cpu_threads,
    )
    if not bool(fit_audit.no_temporal_leakage.all()):
        raise AssertionError("Weighted Ridge training leaked across evaluation origins")
    bands = build_conformal_store(store, base_config["conformal"])
    weighted_cube, weighted_daily, weighted_tuning = decision_analysis(
        store, bands, weighted, base_config["decision"]
    )
    mixed = weighted.copy()
    before = mixed.date.lt(pd.Timestamp("2024-01-01"))
    mixed.loc[before, LOAD_COLUMNS] = original.loc[before, LOAD_COLUMNS].to_numpy()
    fixed_cube, fixed_daily, fixed_tuning = decision_analysis(
        store, bands, mixed, base_config["decision"]
    )
    old_cube = pd.read_csv(
        root / "results" / "ISO_NE_REVISION_V1.0" / "F_nyiso_external" / "nyiso_decision_cube_2025.csv"
    )
    decision_summary = pd.concat([
        primary_decisions(old_cube, "published_equal_observation"),
        primary_decisions(fixed_cube, "duration_weighted_original_threshold"),
        primary_decisions(weighted_cube, "duration_weighted_reestimated_threshold"),
    ], ignore_index=True)

    quality = []
    old_store = pd.read_parquet(
        root / "results" / "ISO_NE_REVISION_V1.0" / "F_nyiso_external" / "nyiso_oos_prediction_store.parquet"
    )
    old_bands = pd.read_parquet(
        root / "results" / "ISO_NE_REVISION_V1.0" / "F_nyiso_external" / "nyiso_conformal_prediction_store.parquet"
    )
    for scenario, current_store, current_bands, current_daily in (
        ("published_equal_observation", old_store, old_bands, original),
        ("duration_weighted", store, bands, weighted),
    ):
        period = current_store.loc[pd.to_datetime(current_store.target_date).between("2025-01-01", "2025-12-31")]
        mae = float(np.mean(np.abs(period[TRUE_COLUMNS].to_numpy(float) - period[PRED_COLUMNS].to_numpy(float))))
        for method in ("standardized_max_90", "one_sided_max_90"):
            row = band_metrics_for_period(
                current_store, current_bands, current_daily, method,
                pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"),
            )
            quality.append({"scenario": scenario, "ridge_mae_mw": mae, **row})
    quality_summary = pd.DataFrame(quality)

    write_csv(archive_hashes, output / "raw_zip_sha256.csv")
    write_csv(long_gaps, output / "long_timestamp_gaps.csv")
    write_csv(hour_check.loc[hour_check.date.between("2025-01-01", "2025-12-31")], output / "hourly_comparison_2025.csv")
    write_csv(comparison.loc[comparison.date.between("2025-01-01", "2025-12-31")], output / "daily_profile_comparison_2025.csv")
    write_csv(events.loc[events.date.between("2024-01-01", "2025-12-31")], output / "event_classification_2024_2025.csv")
    write_csv(decision_summary, output / "primary_decision_comparison.csv")
    write_csv(quality_summary, output / "forecast_band_comparison.csv")
    write_csv(fit_audit, output / "weighted_monthly_fit_audit.csv")
    write_csv(weighted_tuning, output / "weighted_decision_tuning_2024.csv")
    write_csv(fixed_tuning, output / "fixed_threshold_decision_tuning_2024.csv")
    write_csv(weighted_cube, output / "weighted_decision_cube_2025.csv")
    write_parquet(weighted, output / "weighted_daily_profile.parquet")
    write_parquet(features, output / "weighted_features.parquet")
    write_parquet(store, output / "weighted_oos_prediction_store.parquet")
    write_parquet(bands, output / "weighted_conformal_prediction_store.parquet")
    write_parquet(weighted_daily, output / "weighted_decision_daily_2025.parquet")
    write_parquet(fixed_daily, output / "fixed_threshold_decision_daily_2025.parquet")

    y2025 = comparison.loc[comparison.date.between("2025-01-01", "2025-12-31")]
    e2024 = events.loc[events.date.between("2024-01-01", "2024-12-31")]
    e2025 = events.loc[events.date.between("2025-01-01", "2025-12-31")]
    h2025 = hour_check.loc[hour_check.date.between("2025-01-01", "2025-12-31")]
    changed_fixed = e2025.loc[e2025.original_event.ne(e2025.duration_fixed_threshold_event), "date"]
    changed_reestimated = e2025.loc[e2025.original_event.ne(e2025.duration_reestimated_threshold_event), "date"]
    report = f"""# E30 NYISO duration-weighted aggregation sensitivity

This retrospective sensitivity was defined after inspection of the original 2025 results. It does not replace the frozen equal-observation target or establish the operator's physical hourly energy. Each system-load observation is held forward until the next source timestamp, with intervals longer than 30 minutes left unbridged. UTC hours are integrated over elapsed seconds, then converted to local date/hour; the original 23/25-hour DST normalization is reused. At the dataset and long-gap boundaries, at most five minutes are filled with the adjacent observation.

## Input and invariance checks

- Original profiles regenerated from the current 96 raw ZIPs differ from the frozen E0 profiles by at most {discrepancy:.10g} MW.
- Alternative daily availability: {int(weighted.available.sum())} days; DST-adjusted days: {len(dst)}; retained feature dates: {len(features)}.
- Regular 12-observation on-grid UTC hours: {len(regular_check)}; maximum alternative-minus-equal mean difference: {regular_error:.10g} MW.
- Long inter-observation gaps above 30 minutes: {len(long_gaps)}; see `long_timestamp_gaps.csv`.

## 2025 load and event sensitivity

- UTC hours compared: {len(h2025)}; hours with absolute difference above 1e-6 MW: {int((h2025.delta_mw.abs() > 1e-6).sum())}.
- Days with at least one changed hourly position: {int((y2025.max_hour_absolute_delta_mw > 1e-6).sum())} of 365.
- Absolute hourly difference median / p95 / maximum: {h2025.delta_mw.abs().median():.4f} / {h2025.delta_mw.abs().quantile(.95):.4f} / {h2025.delta_mw.abs().max():.4f} MW.
- Absolute daily-peak difference median / p95 / maximum: {y2025.peak_delta_mw.abs().median():.4f} / {y2025.peak_delta_mw.abs().quantile(.95):.4f} / {y2025.peak_delta_mw.abs().max():.4f} MW.
- Historical pre-2024 Q95: equal-observation {original_threshold:.4f} MW; duration-weighted {weighted_threshold:.4f} MW.
- Original 2024 validation events: {int(e2024.original_event.sum())}; changed validation event dates at fixed threshold: {int(e2024.original_event.ne(e2024.duration_fixed_threshold_event).sum())}.
- Original 2025 events: {int(e2025.original_event.sum())}; duration-weighted events at original threshold: {int(e2025.duration_fixed_threshold_event.sum())}; at reestimated threshold: {int(e2025.duration_reestimated_threshold_event.sum())}.
- Original peak-event days with an altered daily peak: {int((e2025.loc[e2025.original_event, 'peak_delta_mw'].abs() > 1e-6).sum())} of {int(e2025.original_event.sum())}; maximum absolute peak change on those days: {e2025.loc[e2025.original_event, 'peak_delta_mw'].abs().max():.4f} MW.
- Event dates changed relative to original at fixed threshold: {', '.join(changed_fixed.dt.strftime('%Y-%m-%d')) or 'none'}.
- Event dates changed relative to original at reestimated threshold: {', '.join(changed_reestimated.dt.strftime('%Y-%m-%d')) or 'none'}.

## Forecast and decision results

All alternative features, monthly Ridge fits, residual bands, and 2024 decision tuning were regenerated from the alternative target. No 2025 target was used in training or parameter selection for its own forecast. `forecast_band_comparison.csv` and `primary_decision_comparison.csv` contain the full numerical comparison. The original threshold is evaluated separately from a pre-2024 reestimated threshold so target sensitivity and threshold sensitivity are not conflated.

```text
{quality_summary[["scenario", "method", "ridge_mae_mw", "daily_simultaneous_coverage_pct", "daily_upper_coverage_pct", "mean_width_mw"]].to_string(index=False)}
```

```text
{decision_summary.to_string(index=False)}
```

The forward-hold calculation is one explicit sensitivity definition. Source timestamps do not independently certify whether an observation represents the start, end, or midpoint of a physical interval. Thus a small difference supports robustness to this specified alternative, not proof of duration-weighted physical truth. A large difference requires updating the manuscript's NYISO interpretation rather than selecting a favorable target.
"""
    (output / "E30_REPORT.md").write_text(report, encoding="utf-8")
    records = [{"relative_path": path.name, "sha256": sha256_bytes(path), "bytes": path.stat().st_size}
               for path in sorted(output.iterdir()) if path.is_file() and path.name != "OUTPUT_SHA256.csv"]
    write_csv(pd.DataFrame(records), output / "OUTPUT_SHA256.csv")


if __name__ == "__main__":
    import argparse

    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output", type=Path, default=project_root / "results" / "ROUND3_MINOR_REVISION_V1" / "E30")
    parser.add_argument("--cpu-threads", type=int, default=12)
    args = parser.parse_args()
    run(args.downloads, args.output, project_root, args.cpu_threads)
