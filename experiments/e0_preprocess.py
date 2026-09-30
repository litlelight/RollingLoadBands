from __future__ import annotations

import argparse
import calendar
import hashlib
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "work" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


ISO_START = pd.Timestamp("2003-01-01")
ISO_END = pd.Timestamp("2026-06-30")
NYISO_START = pd.Timestamp("2018-01-01")
NYISO_END = pd.Timestamp("2025-12-31")
NYISO_GAP_START = pd.Timestamp("2021-04-01")
NYISO_GAP_END = pd.Timestamp("2021-04-07")

PROFILE_LAGS = (1, 2, 7, 14, 21, 28)
SUMMARY_LAGS = (1, 7, 28)
ROLLING_WINDOWS = (7, 14, 28, 56)
SUMMARY_NAMES = ("mean", "max", "min", "range", "std", "ramp_up", "ramp_down")
ROLLING_STD_SUMMARIES = ("mean", "max", "range")

NYISO_ZONES = (
    "CAPITL",
    "CENTRL",
    "DUNWOD",
    "GENESE",
    "HUD VL",
    "LONGIL",
    "MHK VL",
    "MILLWD",
    "N.Y.C.",
    "NORTH",
    "WEST",
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    category: str
    source_date_rule: str
    definition: str
    units: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for lag in PROFILE_LAGS:
        for hour in range(1, 25):
            specs.append(
                FeatureSpec(
                    f"load_lag{lag}_h{hour:02d}",
                    "lagged_load_profile",
                    f"d-{lag}",
                    f"EEI/profile position {hour:02d} from calendar day d-{lag}",
                    "MW",
                )
            )
    for lag in SUMMARY_LAGS:
        for summary in SUMMARY_NAMES:
            specs.append(
                FeatureSpec(
                    f"summary_{summary}_lag{lag}",
                    "lagged_daily_summary",
                    f"d-{lag}",
                    f"Daily {summary.replace('_', ' ')} from calendar day d-{lag}",
                    "MW",
                )
            )
    for window in ROLLING_WINDOWS:
        for summary in SUMMARY_NAMES:
            specs.append(
                FeatureSpec(
                    f"summary_{summary}_rollmean{window}",
                    "rolling_summary_mean",
                    f"d-{window}:d-1",
                    f"Mean of daily {summary.replace('_', ' ')} over the preceding {window} calendar days",
                    "MW",
                )
            )
    for window in ROLLING_WINDOWS:
        for summary in ROLLING_STD_SUMMARIES:
            specs.append(
                FeatureSpec(
                    f"summary_{summary}_rollstd{window}",
                    "rolling_summary_std",
                    f"d-{window}:d-1",
                    f"Sample standard deviation of daily {summary.replace('_', ' ')} over the preceding {window} calendar days",
                    "MW",
                )
            )

    calendar_specs = (
        ("cal_doy_sin", "target d", "Sine encoding of day of year", "unitless"),
        ("cal_doy_cos", "target d", "Cosine encoding of day of year", "unitless"),
        ("cal_dow_sin", "target d", "Sine encoding of day of week", "unitless"),
        ("cal_dow_cos", "target d", "Cosine encoding of day of week", "unitless"),
        ("cal_weekend", "target d", "Saturday or Sunday indicator", "binary"),
        ("cal_federal_holiday", "target d", "US federal holiday or observed-holiday indicator", "binary"),
        ("cal_day_before_holiday", "target d and d+1 calendar", "Target day immediately precedes a US federal holiday", "binary"),
        ("cal_day_after_holiday", "target d and d-1 calendar", "Target day immediately follows a US federal holiday", "binary"),
        ("cal_leap_year", "target d", "Leap-year indicator", "binary"),
    )
    for name, rule, definition, units in calendar_specs:
        specs.append(FeatureSpec(name, "calendar", rule, definition, units))
    for dow in range(7):
        specs.append(
            FeatureSpec(
                f"cal_dow_{dow}",
                "calendar_one_hot",
                "target d",
                f"One-hot day-of-week indicator where Monday=0; category {dow}",
                "binary",
            )
        )
    for month in range(1, 13):
        specs.append(
            FeatureSpec(
                f"cal_month_{month:02d}",
                "calendar_one_hot",
                "target d",
                f"One-hot month indicator for month {month:02d}",
                "binary",
            )
        )
    assert len(specs) == 233, len(specs)
    assert len({spec.name for spec in specs}) == 233
    return specs


def calendar_feature_values(target: pd.Timestamp, holidays: set[pd.Timestamp]) -> dict[str, float]:
    day = pd.Timestamp(target).normalize()
    doy = day.dayofyear
    days_in_year = 366 if calendar.isleap(day.year) else 365
    dow = day.dayofweek
    values: dict[str, float] = {
        "cal_doy_sin": math.sin(2.0 * math.pi * (doy - 1) / days_in_year),
        "cal_doy_cos": math.cos(2.0 * math.pi * (doy - 1) / days_in_year),
        "cal_dow_sin": math.sin(2.0 * math.pi * dow / 7.0),
        "cal_dow_cos": math.cos(2.0 * math.pi * dow / 7.0),
        "cal_weekend": float(dow >= 5),
        "cal_federal_holiday": float(day in holidays),
        "cal_day_before_holiday": float(day + pd.Timedelta(days=1) in holidays),
        "cal_day_after_holiday": float(day - pd.Timedelta(days=1) in holidays),
        "cal_leap_year": float(calendar.isleap(day.year)),
    }
    values.update({f"cal_dow_{idx}": float(dow == idx) for idx in range(7)})
    values.update({f"cal_month_{month:02d}": float(day.month == month) for month in range(1, 13)})
    return values


def daily_summaries(loads: np.ndarray) -> dict[str, float]:
    diffs = np.diff(loads)
    return {
        "mean": float(np.mean(loads)),
        "max": float(np.max(loads)),
        "min": float(np.min(loads)),
        "range": float(np.max(loads) - np.min(loads)),
        "std": float(np.std(loads, ddof=0)),
        "ramp_up": float(np.max(diffs)),
        "ramp_down": float(np.min(diffs)),
    }


def build_model_features(
    daily: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    load_cols = [f"load_h{hour:02d}" for hour in range(1, 25)]
    indexed = daily.set_index("date").sort_index()
    profile_map: dict[pd.Timestamp, np.ndarray] = {}
    summary_map: dict[pd.Timestamp, dict[str, float]] = {}
    for day, row in indexed.iterrows():
        values = row[load_cols].to_numpy(dtype=float)
        if bool(row.get("available", True)) and np.isfinite(values).all():
            day_key = pd.Timestamp(day).normalize()
            profile_map[day_key] = values
            summary_map[day_key] = daily_summaries(values)

    holiday_index = USFederalHolidayCalendar().holidays(
        start=start - pd.Timedelta(days=2),
        end=end + pd.Timedelta(days=2),
    )
    holiday_set = {pd.Timestamp(value).normalize() for value in holiday_index}
    specs = feature_specs()
    feature_names = [spec.name for spec in specs]
    feature_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []

    for target in pd.date_range(start, end, freq="D"):
        target = pd.Timestamp(target).normalize()
        history_dates = [target - pd.Timedelta(days=delta) for delta in range(1, 57)]
        missing_history = [value for value in history_dates if value not in profile_map]
        target_available = target in profile_map
        if not target_available or missing_history:
            if target < start + pd.Timedelta(days=56):
                reason = "initial_56_day_warmup"
            elif not target_available:
                reason = "target_profile_unavailable"
            else:
                reason = "incomplete_56_day_history_after_gap"
            invalid_rows.append(
                {
                    "dataset": dataset,
                    "target_date": target,
                    "reason": reason,
                    "target_available": target_available,
                    "missing_history_count": len(missing_history),
                    "first_missing_history_date": min(missing_history) if missing_history else pd.NaT,
                    "last_missing_history_date": max(missing_history) if missing_history else pd.NaT,
                }
            )
            continue

        row: dict[str, object] = {"date": target}
        for lag in PROFILE_LAGS:
            values = profile_map[target - pd.Timedelta(days=lag)]
            for hour, value in enumerate(values, start=1):
                row[f"load_lag{lag}_h{hour:02d}"] = float(value)
        for lag in SUMMARY_LAGS:
            values = summary_map[target - pd.Timedelta(days=lag)]
            for summary in SUMMARY_NAMES:
                row[f"summary_{summary}_lag{lag}"] = values[summary]
        for window in ROLLING_WINDOWS:
            dates = [target - pd.Timedelta(days=delta) for delta in range(window, 0, -1)]
            for summary in SUMMARY_NAMES:
                values = np.array([summary_map[value][summary] for value in dates], dtype=float)
                row[f"summary_{summary}_rollmean{window}"] = float(np.mean(values))
            for summary in ROLLING_STD_SUMMARIES:
                values = np.array([summary_map[value][summary] for value in dates], dtype=float)
                row[f"summary_{summary}_rollstd{window}"] = float(np.std(values, ddof=1))
        row.update(calendar_feature_values(target, holiday_set))
        target_values = profile_map[target]
        for hour, value in enumerate(target_values, start=1):
            row[f"target_load_h{hour:02d}"] = float(value)
        feature_rows.append(row)

    features = pd.DataFrame(feature_rows)
    invalid = pd.DataFrame(invalid_rows)
    target_names = [f"target_load_h{hour:02d}" for hour in range(1, 25)]
    expected_columns = ["date", *feature_names, *target_names]
    if list(features.columns) != expected_columns:
        features = features.reindex(columns=expected_columns)
    assert features[feature_names].notna().all().all()
    assert features[target_names].notna().all().all()
    return features, invalid


def parse_iso_ne(downloads: Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    archives = (downloads / "my_reports.zip", downloads / "my_reports (1).zip")
    for path in archives:
        if not path.exists():
            raise FileNotFoundError(path)

    audit: list[dict[str, object]] = []
    day_parts: dict[pd.Timestamp, dict[int, tuple[list[float], str, int]]] = {}
    malformed_count = 0
    repair_value = -1

    with zipfile.ZipFile(archives[0]) as original_zip:
        full_2015_lines = original_zip.read("2015_eei_loads.txt").decode("ascii").splitlines()
        full_2015_keys = {
            line[:20].strip(): line
            for line in full_2015_lines
            if len(line) >= 80 and line[:20].strip()
        }

    for archive in archives:
        with zipfile.ZipFile(archive) as zipped:
            bad_member = zipped.testzip()
            audit.append(
                {
                    "issue_id": f"ISO-ARCHIVE-{archive.stem}",
                    "issue_type": "archive_integrity",
                    "source_file": archive.name,
                    "date_or_time": "",
                    "count": 0 if bad_member is None else 1,
                    "action": "none",
                    "status": "PASS" if bad_member is None else "FAIL",
                    "detail": f"sha256={sha256(archive)}" + ("" if bad_member is None else f"; bad_member={bad_member}"),
                }
            )
            for entry in zipped.infolist():
                match = re.fullmatch(r"(\d{4})_eei_loads\.txt", Path(entry.filename).name)
                if not match:
                    continue
                year = int(match.group(1))
                lines = zipped.read(entry).decode("ascii").splitlines()
                for line_number, line in enumerate(lines, start=1):
                    if not line.strip():
                        continue
                    key = line[:20].strip()
                    valid_shape = len(line) >= 80 and len(line[20:]) >= 60 and key and key[-1:] in {"1", "2"}
                    if not valid_shape:
                        confirmed_in_2015 = key in full_2015_keys and len(full_2015_keys[key]) >= 80
                        if year == 2014 and key == "0101151" and confirmed_in_2015:
                            malformed_count += 1
                            audit.append(
                                {
                                    "issue_id": "ISO-MALFORMED-2014-TRAILER",
                                    "issue_type": "malformed_trailing_row",
                                    "source_file": entry.filename,
                                    "date_or_time": "2015-01-01 part 1",
                                    "count": 1,
                                    "action": "removed_from_processed_data",
                                    "status": "PASS",
                                    "detail": "Incomplete trailing key 0101151; complete matching row exists in 2015_eei_loads.txt",
                                }
                            )
                            continue
                        raise ValueError(f"Malformed ISO-NE row: {archive.name}:{entry.filename}:{line_number}: {line!r}")

                    # EEI archives use fixed-width month/day fields. Older files
                    # blank-pad single digits (for example `` 1 4081``), while
                    # newer files use a four-digit year. The filename is the
                    # authoritative year; only month, day, and half-day marker
                    # are parsed from the row prefix.
                    month = int(line[0:2].strip())
                    day = int(line[2:4].strip())
                    if line[8:9] in {"1", "2"}:
                        part = int(line[8])
                    elif line[6:7] in {"1", "2"}:
                        part = int(line[6])
                    else:
                        raise ValueError(f"Cannot parse ISO-NE half-day marker: {entry.filename}:{line_number}")
                    target_date = pd.Timestamp(year=year, month=month, day=day)
                    values = [float(line[20 + idx * 5 : 25 + idx * 5]) for idx in range(12)]
                    if len(values) != 12 or not np.isfinite(values).all():
                        raise ValueError(f"Invalid ISO-NE values: {entry.filename}:{line_number}")
                    if target_date in day_parts and part in day_parts[target_date]:
                        raise ValueError(f"Duplicate ISO-NE day part: {target_date.date()} part {part}")
                    day_parts.setdefault(target_date, {})[part] = (values, entry.filename, line_number)

    rows: list[dict[str, object]] = []
    for target_date in sorted(day_parts):
        parts = day_parts[target_date]
        if set(parts) != {1, 2}:
            raise ValueError(f"ISO-NE date lacks two parts: {target_date.date()} has {sorted(parts)}")
        values = np.array(parts[1][0] + parts[2][0], dtype=float)
        has_repair = False
        repair_reason = ""
        if target_date == pd.Timestamp("2015-03-08"):
            original = values[1]
            if original != 0:
                raise AssertionError(f"Expected 2015-03-08 HE02=0, found {original}")
            repaired = (values[0] + values[2]) / 2.0
            if repaired != 12378:
                raise AssertionError(f"Expected repaired value 12378, found {repaired}")
            values[1] = repaired
            repair_value = int(repaired)
            has_repair = True
            repair_reason = "DST zero placeholder"
            audit.append(
                {
                    "issue_id": "ISO-DST-2015-03-08-HE02",
                    "issue_type": "non_positive_dst_placeholder",
                    "source_file": parts[1][1],
                    "date_or_time": "2015-03-08 HE02",
                    "count": 1,
                    "action": "replaced_with_adjacent_position_mean",
                    "status": "PASS",
                    "detail": f"original=0; h01={int(values[0])}; h03={int(values[2])}; repaired={int(repaired)}",
                }
            )
        row: dict[str, object] = {
            "date": target_date,
            "source_file": parts[1][1],
            "raw_row_id": f"{parts[1][2]}|{parts[2][2]}",
            "available": True,
            "has_repair": has_repair,
            "repair_reason": repair_reason,
        }
        row.update({f"load_h{hour:02d}": value for hour, value in enumerate(values, start=1)})
        rows.append(row)

    daily = pd.DataFrame(rows)
    daily = daily[(daily["date"] >= ISO_START) & (daily["date"] <= ISO_END)].copy()
    daily = daily.sort_values("date").reset_index(drop=True)
    expected_dates = pd.date_range(ISO_START, ISO_END, freq="D")
    actual_dates = pd.DatetimeIndex(daily["date"])
    missing_dates = expected_dates.difference(actual_dates)
    extra_dates = actual_dates.difference(expected_dates)
    load_cols = [f"load_h{hour:02d}" for hour in range(1, 25)]
    non_positive_after = int((daily[load_cols] <= 0).sum().sum())
    checks = (
        ("ISO-DATE-RANGE", "date_range", len(daily), len(daily) == 8582, "2003-01-01 through 2026-06-30"),
        ("ISO-CONTINUITY", "date_continuity", len(missing_dates), len(missing_dates) == 0 and len(extra_dates) == 0, f"missing={len(missing_dates)}; extra={len(extra_dates)}"),
        ("ISO-POSITIONS", "positions_per_day", 24, daily[load_cols].notna().all().all(), "Every processed day has 24 positions"),
        ("ISO-NONPOSITIVE", "non_positive_after_repair", non_positive_after, non_positive_after == 0, "No non-positive processed loads"),
        ("ISO-MALFORMED-COUNT", "malformed_rows_removed", malformed_count, malformed_count == 1, "Expected exactly the confirmed 2014 trailing row"),
    )
    for issue_id, issue_type, count, passed, detail in checks:
        audit.append(
            {
                "issue_id": issue_id,
                "issue_type": issue_type,
                "source_file": "processed ISO-NE",
                "date_or_time": "",
                "count": count,
                "action": "asserted",
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
        )
    if not all(item[3] for item in checks):
        raise AssertionError("ISO-NE audit checks failed")
    return daily, pd.DataFrame(audit), repair_value


def canonical_nyiso_archives(downloads: Path) -> list[Path]:
    archives: list[Path] = []
    for path in downloads.glob("*pal_csv.zip"):
        if re.fullmatch(r"\d{8}pal_csv\.zip", path.name):
            month = path.name[:6]
            if "201801" <= month <= "202512":
                archives.append(path)
    archives.sort(key=lambda value: value.name)
    months = [path.name[:6] for path in archives]
    expected = [value.strftime("%Y%m") for value in pd.date_range(NYISO_START, NYISO_END, freq="MS")]
    if months != expected:
        missing = sorted(set(expected) - set(months))
        extras = sorted(set(months) - set(expected))
        raise AssertionError(f"NYISO monthly coverage mismatch: missing={missing}; extras={extras}")
    return archives


def interpolate_short_gaps(
    pivot: pd.DataFrame,
    meta: pd.DataFrame,
    source_archive: str,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    imputation_log: list[dict[str, object]] = []
    missing_group_log: list[dict[str, object]] = []
    meta_by_utc = meta.set_index("timestamp_utc")
    blank_groups = pivot.isna().any(axis=1)
    for timestamp_utc in pivot.index[blank_groups]:
        missing_zones = [zone for zone in NYISO_ZONES if pd.isna(pivot.at[timestamp_utc, zone])]
        local = meta_by_utc.at[timestamp_utc, "timestamp_local"]
        timezone = meta_by_utc.at[timestamp_utc, "timezone"]
        missing_group_log.append(
            {
                "issue_type": "short_missing_timestamp",
                "source_file": source_archive,
                "date_or_time": f"{pd.Timestamp(local):%Y-%m-%d %H:%M:%S} {timezone}",
                "count": len(missing_zones),
                "action": "zone_wise_linear_time_interpolation_if_gap_le_30_minutes",
                "status": "PENDING",
                "detail": "missing_zones=" + "|".join(missing_zones),
            }
        )

    timestamps = pivot.index
    for zone in NYISO_ZONES:
        values = pivot[zone].to_numpy(dtype=float, copy=True)
        missing_positions = np.flatnonzero(np.isnan(values))
        for position in missing_positions:
            prev_position = position - 1
            while prev_position >= 0 and np.isnan(values[prev_position]):
                prev_position -= 1
            next_position = position + 1
            while next_position < len(values) and np.isnan(values[next_position]):
                next_position += 1
            if prev_position < 0 or next_position >= len(values):
                continue
            prev_time = pd.Timestamp(timestamps[prev_position])
            next_time = pd.Timestamp(timestamps[next_position])
            target_time = pd.Timestamp(timestamps[position])
            enclosing_minutes = (next_time - prev_time).total_seconds() / 60.0
            if enclosing_minutes > 30.0:
                continue
            fraction = (target_time - prev_time).total_seconds() / (next_time - prev_time).total_seconds()
            imputed = float(values[prev_position] + fraction * (values[next_position] - values[prev_position]))
            values[position] = imputed
            local = meta_by_utc.at[target_time, "timestamp_local"]
            timezone = meta_by_utc.at[target_time, "timezone"]
            imputation_log.append(
                {
                    "timestamp_utc": target_time,
                    "timestamp_local": pd.Timestamp(local),
                    "timezone": timezone,
                    "zone": zone,
                    "source_file": source_archive,
                    "previous_timestamp_utc": prev_time,
                    "next_timestamp_utc": next_time,
                    "previous_load_mw": float(values[prev_position]),
                    "next_load_mw": float(values[next_position]),
                    "imputed_load_mw": imputed,
                    "gap_enclosing_minutes": enclosing_minutes,
                    "is_imputed": True,
                    "imputation_method": "linear_time_interpolation",
                }
            )
        pivot[zone] = values

    remaining = int(pivot[list(NYISO_ZONES)].isna().sum().sum())
    imputed_counts: dict[pd.Timestamp, int] = {}
    for item in imputation_log:
        timestamp = pd.Timestamp(item["timestamp_utc"])
        imputed_counts[timestamp] = imputed_counts.get(timestamp, 0) + 1
    for item in missing_group_log:
        local_text, timezone = str(item["date_or_time"]).rsplit(" ", 1)
        local = pd.Timestamp(local_text)
        utc = (local + pd.Timedelta(hours=4 if timezone == "EDT" else 5)).tz_localize("UTC")
        count = imputed_counts.get(utc, 0)
        item["status"] = "PASS" if count == item["count"] else "FAIL"
        item["detail"] += f"; imputed_zone_cells={count}"
    if remaining:
        raise AssertionError(f"NYISO short-gap interpolation left {remaining} missing cells")
    return pivot, imputation_log, missing_group_log


def normalize_nyiso_days(
    hourly: pd.DataFrame,
    imputation_log: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputed_by_date = (
        imputation_log.assign(date=imputation_log["timestamp_local"].dt.normalize())
        .groupby("date")
        .size()
        .to_dict()
        if not imputation_log.empty
        else {}
    )
    rows: list[dict[str, object]] = []
    dst_rows: list[dict[str, object]] = []
    for target_date, group in hourly.groupby("date", sort=True):
        group = group.sort_values(["hour", "timezone"])
        grouped_hours = {int(hour): part for hour, part in group.groupby("hour", sort=True)}
        unique_hours = sorted(grouped_hours)
        total_hour_groups = len(group)
        dst_adjusted = False
        dst_type = ""
        values: dict[int, float] = {}
        if unique_hours == list(range(24)) and total_hour_groups == 24:
            for hour, part in grouped_hours.items():
                values[hour] = float(part["system_load_mw"].iloc[0])
        elif len(unique_hours) == 23 and total_hour_groups == 23:
            missing_hours = sorted(set(range(24)) - set(unique_hours))
            if missing_hours != [2]:
                raise AssertionError(f"Unexpected 23-hour day {target_date}: missing {missing_hours}")
            for hour, part in grouped_hours.items():
                values[hour] = float(part["system_load_mw"].iloc[0])
            values[2] = (values[1] + values[3]) / 2.0
            dst_adjusted = True
            dst_type = "spring_forward"
            dst_rows.append(
                {
                    "date": pd.Timestamp(target_date),
                    "dst_type": dst_type,
                    "raw_hour_groups": total_hour_groups,
                    "affected_local_hour": 2,
                    "timezones": "|".join(sorted(group["timezone"].unique())),
                    "action": "interpolated_missing_hour_from_adjacent_hourly_means",
                    "normalized_positions": 24,
                }
            )
        elif unique_hours == list(range(24)) and total_hour_groups == 25:
            repeated = [hour for hour, part in grouped_hours.items() if len(part) == 2]
            if repeated != [1]:
                raise AssertionError(f"Unexpected 25-hour day {target_date}: repeated {repeated}")
            for hour, part in grouped_hours.items():
                values[hour] = float(part["system_load_mw"].mean())
            dst_adjusted = True
            dst_type = "fall_back"
            dst_rows.append(
                {
                    "date": pd.Timestamp(target_date),
                    "dst_type": dst_type,
                    "raw_hour_groups": total_hour_groups,
                    "affected_local_hour": 1,
                    "timezones": "|".join(sorted(group["timezone"].unique())),
                    "action": "averaged_repeated_EDT_and_EST_hourly_means",
                    "normalized_positions": 24,
                }
            )
        else:
            raise AssertionError(
                f"NYISO day cannot be normalized: {target_date}; unique_hours={unique_hours}; groups={total_hour_groups}"
            )
        if set(values) != set(range(24)):
            raise AssertionError(f"NYISO normalized day lacks hours: {target_date}")
        row: dict[str, object] = {
            "date": pd.Timestamp(target_date),
            "source_file": "|".join(sorted(group["source_archive"].unique())),
            "available": True,
            "gap_reason": "",
            "has_imputation": pd.Timestamp(target_date) in imputed_by_date,
            "imputed_zone_cell_count": int(imputed_by_date.get(pd.Timestamp(target_date), 0)),
            "dst_adjusted": dst_adjusted,
            "dst_type": dst_type,
        }
        row.update({f"load_h{hour + 1:02d}": values[hour] for hour in range(24)})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(dst_rows)


def parse_nyiso(downloads: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    archives = canonical_nyiso_archives(downloads)
    audit: list[dict[str, object]] = []
    hourly_frames: list[pd.DataFrame] = []
    imputation_rows: list[dict[str, object]] = []
    missing_group_rows: list[dict[str, object]] = []
    irregular_timestamp_count = 0
    raw_row_count = 0

    expected_months = pd.date_range(NYISO_START, NYISO_END, freq="MS")
    for archive, month_start in zip(archives, expected_months, strict=True):
        frames: list[pd.DataFrame] = []
        with zipfile.ZipFile(archive) as zipped:
            bad_member = zipped.testzip()
            if bad_member is not None:
                raise ValueError(f"Damaged NYISO archive {archive.name}: {bad_member}")
            member_map = {Path(item.filename).name: item for item in zipped.infolist() if item.filename.lower().endswith(".csv")}
            expected_days = pd.date_range(month_start, month_start + pd.offsets.MonthEnd(0), freq="D")
            expected_members = {f"{value:%Y%m%d}pal.csv" for value in expected_days}
            missing_members = sorted(expected_members - set(member_map))
            for member in missing_members:
                missing_date = pd.Timestamp(datetime.strptime(member[:8], "%Y%m%d"))
                expected_gap = NYISO_GAP_START <= missing_date <= NYISO_GAP_END
                audit.append(
                    {
                        "issue_type": "official_archive_gap_day" if expected_gap else "missing_daily_member",
                        "source_file": archive.name,
                        "date_or_time": f"{missing_date:%Y-%m-%d}",
                        "count": 1,
                        "action": "excluded; no cross-gap interpolation",
                        "status": "PASS" if expected_gap else "FAIL",
                        "detail": "P-58B official archive gap" if expected_gap else "Unexpected missing daily CSV member",
                    }
                )
            unexpected = sorted(set(member_map) - expected_members)
            if unexpected:
                raise AssertionError(f"Unexpected NYISO members in {archive.name}: {unexpected[:5]}")

            for member_name in sorted(member_map):
                with zipped.open(member_map[member_name]) as handle:
                    raw = pd.read_csv(handle)
                required = ["Time Stamp", "Time Zone", "Name", "PTID", "Load"]
                if list(raw.columns) != required:
                    raise ValueError(f"Unexpected NYISO header in {archive.name}/{member_name}: {list(raw.columns)}")
                frame = raw.rename(
                    columns={
                        "Time Stamp": "timestamp_local",
                        "Time Zone": "timezone",
                        "Name": "zone",
                        "PTID": "ptid",
                        "Load": "load_mw",
                    }
                )
                frame["timestamp_local"] = pd.to_datetime(
                    frame["timestamp_local"], format="%m/%d/%Y %H:%M:%S", errors="raise"
                )
                frame["timezone"] = frame["timezone"].astype(str)
                if not set(frame["timezone"]).issubset({"EST", "EDT"}):
                    raise AssertionError(f"Unexpected NYISO timezone in {archive.name}/{member_name}")
                frame["zone"] = frame["zone"].astype(str)
                if set(frame["zone"]) != set(NYISO_ZONES):
                    raise AssertionError(f"Unexpected NYISO zones in {archive.name}/{member_name}: {sorted(set(frame['zone']))}")
                frame["load_mw"] = pd.to_numeric(frame["load_mw"], errors="coerce")
                frame["source_archive"] = archive.name
                frame["source_entry"] = member_name
                frames.append(frame)

        if not frames:
            continue
        month = pd.concat(frames, ignore_index=True)
        raw_row_count += len(month)
        offsets = month["timezone"].map({"EDT": 4, "EST": 5})
        utc_naive = month["timestamp_local"] + pd.to_timedelta(offsets, unit="h")
        month["timestamp_utc"] = utc_naive.dt.tz_localize("UTC")
        duplicate_rows = month.duplicated(["timestamp_utc", "zone"], keep=False)
        if duplicate_rows.any():
            sample = month.loc[duplicate_rows, ["timestamp_local", "timezone", "zone"]].head().to_dict("records")
            raise AssertionError(f"Duplicate NYISO UTC-zone keys in {archive.name}: {sample}")
        counts = month.groupby("timestamp_utc")["zone"].nunique()
        if not (counts == 11).all():
            raise AssertionError(f"NYISO timestamp groups without 11 zones in {archive.name}")
        irregular_timestamp_count += int(
            month.loc[
                (month["timestamp_local"].dt.minute % 5 != 0) | (month["timestamp_local"].dt.second != 0),
                "timestamp_utc",
            ].nunique()
        )
        meta = (
            month[["timestamp_utc", "timestamp_local", "timezone", "source_archive"]]
            .drop_duplicates("timestamp_utc")
            .sort_values("timestamp_utc")
        )
        pivot = month.pivot(index="timestamp_utc", columns="zone", values="load_mw").sort_index()
        pivot = pivot.reindex(columns=list(NYISO_ZONES))
        pivot, month_imputations, month_missing_groups = interpolate_short_gaps(pivot, meta, archive.name)
        imputation_rows.extend(month_imputations)
        missing_group_rows.extend(month_missing_groups)
        system = pivot.sum(axis=1, min_count=11).rename("system_load_mw").to_frame().reset_index()
        system = system.merge(meta, on="timestamp_utc", how="left", validate="one_to_one")
        system["date"] = system["timestamp_local"].dt.normalize()
        system["hour"] = system["timestamp_local"].dt.hour
        hourly = (
            system.groupby(["date", "hour", "timezone", "source_archive"], as_index=False)
            .agg(system_load_mw=("system_load_mw", "mean"), raw_sample_count=("system_load_mw", "size"))
        )
        hourly_frames.append(hourly)

    imputation_log = pd.DataFrame(imputation_rows)
    if not imputation_log.empty:
        for column in ("timestamp_utc", "timestamp_local", "previous_timestamp_utc", "next_timestamp_utc"):
            imputation_log[column] = pd.to_datetime(imputation_log[column])
    hourly_all = pd.concat(hourly_frames, ignore_index=True)
    daily_available, dst_adjustments = normalize_nyiso_days(hourly_all, imputation_log)

    calendar_days = pd.DataFrame({"date": pd.date_range(NYISO_START, NYISO_END, freq="D")})
    daily = calendar_days.merge(daily_available, on="date", how="left", validate="one_to_one")
    daily["available"] = daily["available"].fillna(False).astype(bool)
    gap_mask = daily["date"].between(NYISO_GAP_START, NYISO_GAP_END)
    daily.loc[gap_mask, "gap_reason"] = "P58B official archive gap"
    for column in ("source_file", "gap_reason", "dst_type"):
        daily[column] = daily[column].fillna("")
    for column in ("has_imputation", "dst_adjusted"):
        daily[column] = daily[column].fillna(False).astype(bool)
    daily["imputed_zone_cell_count"] = daily["imputed_zone_cell_count"].fillna(0).astype(int)
    load_cols = [f"load_h{hour:02d}" for hour in range(1, 25)]

    missing_group_df = pd.DataFrame(missing_group_rows)
    audit.extend(missing_group_rows)
    audit.extend(
        [
            {
                "issue_type": "monthly_archive_coverage",
                "source_file": "2018-01 through 2025-12",
                "date_or_time": "",
                "count": len(archives),
                "action": "asserted",
                "status": "PASS" if len(archives) == 96 else "FAIL",
                "detail": "Expected 96 canonical monthly archives; duplicate (1) downloads ignored",
            },
            {
                "issue_type": "zone_schema",
                "source_file": "all canonical archives",
                "date_or_time": "",
                "count": len(NYISO_ZONES),
                "action": "asserted",
                "status": "PASS",
                "detail": "|".join(NYISO_ZONES),
            },
            {
                "issue_type": "raw_row_count",
                "source_file": "all canonical archives",
                "date_or_time": "",
                "count": raw_row_count,
                "action": "counted",
                "status": "PASS",
                "detail": "All raw zone-level rows scanned",
            },
            {
                "issue_type": "irregular_non_5min_timestamp",
                "source_file": "all canonical archives",
                "date_or_time": "",
                "count": irregular_timestamp_count,
                "action": "retained_in_hourly_arithmetic_mean",
                "status": "PASS",
                "detail": "Observed source timestamps not exactly aligned to a 5-minute clock were retained and audited",
            },
            {
                "issue_type": "short_missing_timestamp_total",
                "source_file": "all canonical archives",
                "date_or_time": "",
                "count": len(missing_group_df),
                "action": "zone_wise_linear_time_interpolation",
                "status": "PASS" if len(missing_group_df) == 15 else "FAIL",
                "detail": f"imputed_zone_cells={len(imputation_log)}; maximum permitted enclosing gap=30 minutes",
            },
            {
                "issue_type": "dst_normalization",
                "source_file": "all canonical archives",
                "date_or_time": "",
                "count": len(dst_adjustments),
                "action": "spring interpolation; fall repeated-hour averaging",
                "status": "PASS" if len(dst_adjustments) == 16 else "FAIL",
                "detail": "Timezone retained through EDT/EST-aware UTC key before 24-position normalization",
            },
            {
                "issue_type": "available_daily_profiles",
                "source_file": "processed NYISO",
                "date_or_time": "",
                "count": int(daily["available"].sum()),
                "action": "asserted",
                "status": "PASS" if int(daily["available"].sum()) == 2915 else "FAIL",
                "detail": "2922 calendar days minus seven official P-58B archive-gap days",
            },
            {
                "issue_type": "positions_per_available_day",
                "source_file": "processed NYISO",
                "date_or_time": "",
                "count": 24,
                "action": "asserted",
                "status": "PASS" if daily.loc[daily["available"], load_cols].notna().all().all() else "FAIL",
                "detail": "Every available standardized daily profile has 24 positions",
            },
        ]
    )
    audit_df = pd.DataFrame(audit)
    if (audit_df["status"] == "FAIL").any():
        failures = audit_df.loc[audit_df["status"] == "FAIL"].to_dict("records")
        raise AssertionError(f"NYISO audit checks failed: {failures[:5]}")
    return daily, audit_df, dst_adjustments, imputation_log


def write_feature_manifest(path: Path) -> pd.DataFrame:
    specs = feature_specs()
    manifest = pd.DataFrame(
        [
            {
                "feature_index": index,
                "feature_name": spec.name,
                "category": spec.category,
                "source_date_rule": spec.source_date_rule,
                "known_at_forecast_origin": True,
                "definition": spec.definition,
                "units": spec.units,
            }
            for index, spec in enumerate(specs, start=1)
        ]
    )
    manifest.to_csv(path, index=False, encoding="utf-8")
    return manifest


def report_text(
    iso_daily: pd.DataFrame,
    iso_features: pd.DataFrame,
    iso_audit: pd.DataFrame,
    repair_value: int,
    nyiso_daily: pd.DataFrame,
    nyiso_features: pd.DataFrame,
    nyiso_audit: pd.DataFrame,
    nyiso_invalid: pd.DataFrame,
    dst_adjustments: pd.DataFrame,
    imputation_log: pd.DataFrame,
    manifest: pd.DataFrame,
) -> str:
    nyiso_short_missing = int((nyiso_audit["issue_type"] == "short_missing_timestamp").sum())
    irregular_row = nyiso_audit.loc[nyiso_audit["issue_type"] == "irregular_non_5min_timestamp", "count"]
    irregular_timestamps = int(irregular_row.iloc[0]) if len(irregular_row) else 0
    iso_pass = (
        len(iso_daily) == 8582
        and len(iso_features) == 8526
        and len(manifest) == 233
        and repair_value == 12378
        and (iso_audit["status"] == "PASS").all()
    )
    nyiso_pass = (
        len(nyiso_daily) == 2922
        and int(nyiso_daily["available"].sum()) == 2915
        and len(nyiso_features) == 2803
        and len(nyiso_invalid) == 119
        and nyiso_short_missing == 15
        and len(imputation_log) == 165
        and len(dst_adjustments) == 16
        and len(manifest) == 233
        and (nyiso_audit["status"] == "PASS").all()
    )
    first_post_gap = nyiso_features.loc[nyiso_features["date"] > NYISO_GAP_END, "date"].min()
    return f"""# E0 Data Audit and Preprocessing Report

## Scope

This stage performed data ingestion, integrity checks, documented repair or imputation, DST normalization, exact-calendar feature construction, and information-boundary checks. No model was trained and no global predictor standardization was applied.

## ISO-NE

```text
Date range:        {iso_daily['date'].min():%Y-%m-%d} - {iso_daily['date'].max():%Y-%m-%d}
Raw days:          {len(iso_daily)}
Positions/day:     24
Hourly positions:  {len(iso_daily) * 24}
2014 malformed:    removed = {'YES' if (iso_audit['issue_type'] == 'malformed_trailing_row').any() else 'NO'}
2015 DST zero:     found = {'YES' if (iso_audit['issue_type'] == 'non_positive_dst_placeholder').any() else 'NO'}
Repaired value:    {repair_value}
Usable after 56d:  {len(iso_features)}
Feature count:     {len(manifest)}
Status:            {'PASS' if iso_pass else 'FAIL'}
```

The 2014 trailing fragment was removed only after confirming that the complete matching `0101151` record exists in the 2015 source file. The raw 2015-03-08 hour-ending 02 value was asserted to be 0 MW and the processed value was asserted to be 12,378 MW, the mean of its adjacent positions. Raw archives were not modified.

## NYISO

```text
Date range:        {nyiso_daily['date'].min():%Y-%m-%d} - {nyiso_daily['date'].max():%Y-%m-%d}
Calendar days:     {len(nyiso_daily)}
Official gap days: {int((~nyiso_daily['available']).sum())}
Available days:    {int(nyiso_daily['available'].sum())}
Gap:               2021-04-01 - 2021-04-07
First valid target after gap: {first_post_gap:%Y-%m-%d}
Expected full-feature targets: {len(nyiso_features)}
Zones:             {len(NYISO_ZONES)}
Short missing timestamps: {nyiso_short_missing}
Imputed zone cells: {len(imputation_log)}
Non-grid source timestamps: {irregular_timestamps}
DST adjusted days: {len(dst_adjustments)}
DST timezone preserved: YES
Feature count:     {len(manifest)}
Status:            {'PASS' if nyiso_pass else 'FAIL'}
```

The seven-day P-58B gap was not filled or bridged. Complete 56-day feature history is unavailable from 2021-04-01 through 2021-06-02, so the first valid post-gap target is 2021-06-03. Fifteen short timestamp gaps affecting all 11 zones were linearly interpolated zone by zone only when valid observations existed on both sides and the enclosing gap was at most 30 minutes. The source also contains {irregular_timestamps} reported timestamps that are not exactly aligned to a five-minute clock; they were retained and each reported observation received equal weight in its local-hour arithmetic mean. EDT and EST were converted to distinct UTC instants before aggregation. Spring-forward days were normalized by interpolating the missing local hour; fall-back days were normalized by averaging the two complete repeated-hour means.

## Feature boundary

The 233 predictors comprise 205 load-history features and 28 calendar features:

- 144 hourly load-profile values at lags 1, 2, 7, 14, 21, and 28 days.
- 21 lagged daily summaries for lags 1, 7, and 28 days.
- 28 rolling means of seven daily summaries over 7, 14, 28, and 56 days.
- 12 rolling standard deviations for daily mean, maximum, and range over the same four windows.
- 28 target-date calendar variables.

All load lookups use exact calendar dates. Every manifest entry is marked as known at the forecast origin. The saved feature tables contain raw, unstandardized predictors; scaling must be fitted separately inside each future model-training window.

## Gate decision

E0 overall status: **{'PASS' if iso_pass and nyiso_pass else 'FAIL'}**

{'E1 may begin because all mandatory E0 assertions passed.' if iso_pass and nyiso_pass else 'Model training is blocked until every failed E0 assertion is resolved.'}
"""


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    check = pd.read_parquet(path, engine="pyarrow")
    if len(check) != len(frame) or list(check.columns) != list(frame.columns):
        raise AssertionError(f"Parquet round-trip failed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="E0 preprocessing for ISO-NE and NYISO")
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    iso_daily, iso_audit, repair_value = parse_iso_ne(args.downloads)
    nyiso_daily, nyiso_audit, dst_adjustments, imputation_log = parse_nyiso(args.downloads)

    iso_features, iso_invalid = build_model_features(iso_daily, ISO_START, ISO_END, "ISO-NE")
    nyiso_features, nyiso_invalid = build_model_features(nyiso_daily, NYISO_START, NYISO_END, "NYISO")

    if len(iso_invalid) != 56 or len(iso_features) != 8526:
        raise AssertionError(f"Unexpected ISO-NE feature validity: invalid={len(iso_invalid)}, valid={len(iso_features)}")
    if len(nyiso_invalid) != 119 or len(nyiso_features) != 2803:
        raise AssertionError(f"Unexpected NYISO feature validity: invalid={len(nyiso_invalid)}, valid={len(nyiso_features)}")
    if nyiso_features.loc[nyiso_features["date"] > NYISO_GAP_END, "date"].min() != pd.Timestamp("2021-06-03"):
        raise AssertionError("Unexpected first valid NYISO feature target after gap")

    manifest = write_feature_manifest(output / "feature_manifest.csv")
    iso_audit.to_csv(output / "audit_iso_ne.csv", index=False, encoding="utf-8")
    nyiso_audit.to_csv(output / "audit_nyiso.csv", index=False, encoding="utf-8")
    nyiso_invalid.to_csv(output / "invalid_target_dates_nyiso.csv", index=False, encoding="utf-8")
    dst_adjustments.to_csv(output / "dst_adjustments_nyiso.csv", index=False, encoding="utf-8")
    imputation_log.to_csv(output / "imputation_log_nyiso.csv", index=False, encoding="utf-8")

    write_parquet(iso_daily, output / "ISO_NE_daily_profile.parquet")
    write_parquet(nyiso_daily, output / "NYISO_daily_profile.parquet")
    write_parquet(iso_features, output / "ISO_NE_features.parquet")
    write_parquet(nyiso_features, output / "NYISO_features.parquet")

    report = report_text(
        iso_daily,
        iso_features,
        iso_audit,
        repair_value,
        nyiso_daily,
        nyiso_features,
        nyiso_audit,
        nyiso_invalid,
        dst_adjustments,
        imputation_log,
        manifest,
    )
    (output / "data_audit_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
