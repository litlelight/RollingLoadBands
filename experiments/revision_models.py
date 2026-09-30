from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "work" / "pydeps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from experiments.revision_common import (
    LOAD_COLUMNS,
    LOAD_HOURS,
    PRED_COLUMNS,
    TARGET_COLUMNS,
    TRUE_COLUMNS,
    log,
    point_metrics,
    write_csv,
    write_json,
    write_parquet,
)


def _xgb_monthly_predictions(
    features: pd.DataFrame,
    feature_names: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    training_years: int,
    parameters: dict[str, Any],
    device: str,
    random_seed: int,
    cpu_threads: int,
    label: str,
) -> pd.DataFrame:
    dates = pd.to_datetime(features["date"])
    x = features[feature_names].to_numpy(dtype=np.float32)
    y = features[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    predictions = np.full_like(y, np.nan)
    for month_start in pd.date_range(start.replace(day=1), end, freq="MS"):
        train_start = month_start - pd.DateOffset(years=training_years)
        train_end = month_start - pd.Timedelta(days=1)
        test_end = min(month_start + pd.offsets.MonthEnd(0), end)
        train = dates.between(train_start, train_end).to_numpy()
        test = dates.between(max(start, month_start), test_end).to_numpy()
        if not test.any():
            continue
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device=device,
            multi_strategy="one_output_per_tree",
            random_state=random_seed,
            n_jobs=max(1, int(cpu_threads)),
            verbosity=0,
            **{key: value for key, value in parameters.items() if key != "name"},
        )
        model.fit(x[train], y[train])
        predictions[test] = model.predict(x[test])
        if month_start.month in (1, 7):
            log(f"XGBoost {label} {month_start:%Y-%m}: train={int(train.sum())}, predict={int(test.sum())}, device={device}")
    selection = dates.between(start, end).to_numpy()
    if not np.isfinite(predictions[selection]).all():
        raise AssertionError(f"Missing XGBoost predictions for {label}")
    records: dict[str, Any] = {"target_date": dates[selection].to_numpy(), "model": label}
    actual = y[selection].astype(float)
    predicted = predictions[selection].astype(float)
    for hour_index in range(LOAD_HOURS):
        records[TRUE_COLUMNS[hour_index]] = actual[:, hour_index]
        records[PRED_COLUMNS[hour_index]] = predicted[:, hour_index]
    return pd.DataFrame(records)


def _metric_rows(frame: pd.DataFrame, model: str, periods: list[tuple[str, pd.Timestamp, pd.Timestamp]]) -> list[dict[str, Any]]:
    rows = []
    dates = pd.to_datetime(frame["target_date"])
    for label, start, end in periods:
        selected = frame.loc[dates.between(start, end)]
        if not len(selected):
            continue
        rows.append(
            {
                "model": model,
                "period": label,
                "days": len(selected),
                **point_metrics(selected[TRUE_COLUMNS].to_numpy(float), selected[PRED_COLUMNS].to_numpy(float)),
            }
        )
    return rows


def run_xgboost_experiment(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    random_seed: int,
    cpu_threads: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = manifest["feature_name"].tolist()
    development_start = pd.Timestamp(config["monthly_development_start"])
    development_end = pd.Timestamp(config["monthly_development_end"])
    final_start = pd.Timestamp(config["final_prediction_start"])
    final_end = pd.Timestamp(config["final_prediction_end"])
    requested_device = str(config.get("device", "cuda"))
    device = requested_device
    candidate_predictions: dict[str, pd.DataFrame] = {}
    candidate_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for candidate in config["candidates"]:
        name = str(candidate["name"])
        candidate_start = time.perf_counter()
        try:
            frame = _xgb_monthly_predictions(
                features,
                feature_names,
                development_start,
                development_end,
                5,
                candidate,
                device,
                random_seed,
                min(4, cpu_threads),
                f"xgb_{name}",
            )
        except Exception as exc:
            if device == "cpu":
                raise
            log(f"XGBoost GPU failed for {name}; retrying on CPU: {exc}")
            device = "cpu"
            frame = _xgb_monthly_predictions(
                features,
                feature_names,
                development_start,
                development_end,
                5,
                candidate,
                device,
                random_seed,
                cpu_threads,
                f"xgb_{name}",
            )
        candidate_predictions[name] = frame
        values = point_metrics(frame[TRUE_COLUMNS].to_numpy(float), frame[PRED_COLUMNS].to_numpy(float))
        candidate_rows.append(
            {
                "candidate": name,
                "device": device,
                "development_start": development_start,
                "development_end": development_end,
                "days": len(frame),
                "runtime_seconds": time.perf_counter() - candidate_start,
                **candidate,
                **values,
            }
        )
        log(f"XGBoost candidate {name}: development MAE={values['mae_mw']:.3f} MW")

    candidate_metrics = pd.DataFrame(candidate_rows).sort_values(["mae_mw", "n_estimators", "max_depth"])
    selected_name = str(candidate_metrics.iloc[0]["candidate"])
    selected_parameters = next(candidate for candidate in config["candidates"] if candidate["name"] == selected_name)
    final_predictions = _xgb_monthly_predictions(
        features,
        feature_names,
        final_start,
        final_end,
        5,
        selected_parameters,
        device,
        random_seed,
        min(4, cpu_threads) if device == "cuda" else cpu_threads,
        f"xgb_selected_{selected_name}",
    )
    all_predictions = pd.concat(
        [candidate_predictions[selected_name], final_predictions], ignore_index=True
    ).sort_values("target_date").reset_index(drop=True)
    if all_predictions["target_date"].duplicated().any():
        raise AssertionError("Duplicate selected XGBoost prediction dates")
    periods = [
        (str(year), pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
        for year in range(2020, 2026)
    ] + [("2026H1", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-30"))]
    metrics = pd.DataFrame(_metric_rows(all_predictions, f"XGBoost {selected_name}", periods))
    write_csv(candidate_metrics, output_dir / "xgboost_development_candidates.csv")
    write_parquet(all_predictions, output_dir / "xgboost_selected_predictions.parquet")
    write_csv(metrics, output_dir / "xgboost_selected_metrics.csv")
    summary = {
        "status": "PASS",
        "selected_candidate": selected_name,
        "selected_parameters": selected_parameters,
        "device_used": device,
        "development_rule": "lowest 2020-2023 monthly rolling-origin MAE; ties favor fewer trees then shallower depth",
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(summary, output_dir / "xgboost_run_summary.json")
    return summary


def _prepare_tcn_arrays(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    calendar_names: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    available = daily.get("available", pd.Series(True, index=daily.index)).astype(bool)
    profile = daily.loc[available, ["date", *LOAD_COLUMNS]].copy()
    profile["date"] = pd.to_datetime(profile["date"])
    profile_map = {
        pd.Timestamp(row.date): np.asarray([getattr(row, column) for column in LOAD_COLUMNS], dtype=np.float32)
        for row in profile.itertuples(index=False)
    }
    calendar_frame = features.set_index(pd.to_datetime(features["date"]))[calendar_names]
    dates = []
    sequences = []
    calendars = []
    targets = []
    for target in pd.date_range(start, end, freq="D"):
        if target not in profile_map or target not in calendar_frame.index:
            continue
        history = [target - pd.Timedelta(days=lag) for lag in range(lookback, 0, -1)]
        if any(date not in profile_map for date in history):
            continue
        dates.append(target)
        sequences.append(np.stack([profile_map[date] for date in history]))
        calendars.append(calendar_frame.loc[target].to_numpy(dtype=np.float32))
        targets.append(profile_map[target])
    return (
        np.asarray(dates, dtype="datetime64[ns]"),
        np.asarray(sequences, dtype=np.float32),
        np.asarray(calendars, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def run_tcn_experiment(
    daily: pd.DataFrame,
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    output_dir.mkdir(parents=True, exist_ok=True)
    train_start = pd.Timestamp(config["train_start"])
    train_end = pd.Timestamp(config["train_end"])
    validation_start = pd.Timestamp(config["validation_start"])
    validation_end = pd.Timestamp(config["validation_end"])
    evaluation_start = pd.Timestamp(config["evaluation_start"])
    evaluation_end = pd.Timestamp(config["evaluation_end"])
    lookback = int(config["lookback_days"])
    calendar_names = manifest.loc[
        manifest["category"].isin(["calendar", "calendar_one_hot"]), "feature_name"
    ].tolist()
    dates, sequences, calendars, targets = _prepare_tcn_arrays(
        daily,
        features,
        calendar_names,
        train_start,
        evaluation_end,
        lookback,
    )
    timestamps = pd.to_datetime(dates)
    train_mask = timestamps.to_series(index=np.arange(len(timestamps))).between(train_start, train_end).to_numpy()
    validation_mask = timestamps.to_series(index=np.arange(len(timestamps))).between(validation_start, validation_end).to_numpy()
    evaluation_mask = timestamps.to_series(index=np.arange(len(timestamps))).between(evaluation_start, evaluation_end).to_numpy()
    if train_mask.sum() < 1000 or validation_mask.sum() != 366 or evaluation_mask.sum() != 365:
        raise AssertionError(
            f"Unexpected TCN split sizes: train={train_mask.sum()}, validation={validation_mask.sum()}, evaluation={evaluation_mask.sum()}"
        )

    x_mean = sequences[train_mask].mean(axis=(0, 1), keepdims=True)
    x_std = sequences[train_mask].std(axis=(0, 1), keepdims=True)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    y_mean = targets[train_mask].mean(axis=0, keepdims=True)
    y_std = targets[train_mask].std(axis=0, keepdims=True)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    calendar_scaler = StandardScaler().fit(calendars[train_mask])
    x_scaled = ((sequences - x_mean) / x_std).astype(np.float32)
    y_scaled = ((targets - y_mean) / y_std).astype(np.float32)
    calendar_scaled = calendar_scaler.transform(calendars).astype(np.float32)

    class CausalBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            padding = dilation * (kernel_size - 1)
            self.padding = padding
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
            self.norm = nn.GroupNorm(8, out_channels)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        def forward(self, x):
            value = self.conv(x)
            if self.padding:
                value = value[:, :, :-self.padding]
            value = self.dropout(self.activation(self.norm(value)))
            return self.activation(value + self.residual(x))

    class ProfileTCN(nn.Module):
        def __init__(self):
            super().__init__()
            channels = int(config["channels"])
            kernel = int(config["kernel_size"])
            dropout = float(config["dropout"])
            layers = []
            input_channels = LOAD_HOURS
            for dilation in config["dilations"]:
                layers.append(CausalBlock(input_channels, channels, kernel, int(dilation), dropout))
                input_channels = channels
            self.network = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Linear(channels + len(calendar_names), int(config["dense_units"])),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(int(config["dense_units"]), LOAD_HOURS),
            )

        def forward(self, sequence, calendar_values):
            encoded = self.pool(self.network(sequence)).squeeze(-1)
            return self.head(torch.cat([encoded, calendar_values], dim=1))

    use_cuda = bool(torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    log(f"TCN device: {device}; train={int(train_mask.sum())}, validation={int(validation_mask.sum())}, evaluation={int(evaluation_mask.sum())}")

    def tensors(mask: np.ndarray) -> TensorDataset:
        seq = torch.from_numpy(x_scaled[mask]).permute(0, 2, 1).contiguous()
        cal = torch.from_numpy(calendar_scaled[mask])
        y = torch.from_numpy(y_scaled[mask])
        return TensorDataset(seq, cal, y)

    train_dataset = tensors(train_mask)
    val_dataset = tensors(validation_mask)
    eval_sequence = torch.from_numpy(x_scaled[evaluation_mask]).permute(0, 2, 1).contiguous()
    eval_calendar = torch.from_numpy(calendar_scaled[evaluation_mask])
    actual_eval = targets[evaluation_mask].astype(float)
    evaluation_dates = timestamps[evaluation_mask].to_numpy()
    history_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    aggregate_predictions = []
    started = time.perf_counter()

    for seed in [int(value) for value in config["seeds"]]:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=0,
            pin_memory=use_cuda,
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
        )
        model = ProfileTCN().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        loss_function = nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        patience_remaining = int(config["patience"])
        seed_started = time.perf_counter()

        for epoch in range(1, int(config["max_epochs"]) + 1):
            model.train()
            train_losses = []
            for seq_batch, cal_batch, y_batch in train_loader:
                seq_batch = seq_batch.to(device, non_blocking=True)
                cal_batch = cal_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    prediction = model(seq_batch, cal_batch)
                    loss = loss_function(prediction, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_losses.append(float(loss.detach().cpu()))

            model.eval()
            validation_predictions = []
            validation_targets = []
            with torch.no_grad():
                for seq_batch, cal_batch, y_batch in val_loader:
                    seq_batch = seq_batch.to(device, non_blocking=True)
                    cal_batch = cal_batch.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_cuda):
                        predicted_scaled = model(seq_batch, cal_batch)
                    validation_predictions.append(predicted_scaled.float().cpu().numpy())
                    validation_targets.append(y_batch.numpy())
            val_pred = np.vstack(validation_predictions) * y_std + y_mean
            val_true = np.vstack(validation_targets) * y_std + y_mean
            val_mae = float(np.mean(np.abs(val_true - val_pred)))
            history_rows.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "train_mse_standardized": float(np.mean(train_losses)),
                    "validation_mae_mw": val_mae,
                }
            )
            if val_mae < best_loss - 1e-4:
                best_loss = val_mae
                best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                best_epoch = epoch
                patience_remaining = int(config["patience"])
            else:
                patience_remaining -= 1
            if epoch == 1 or epoch % 5 == 0:
                log(f"TCN seed={seed} epoch={epoch}: validation MAE={val_mae:.3f} MW")
            if patience_remaining <= 0:
                break

        if best_state is None:
            raise AssertionError(f"TCN seed {seed} did not produce a checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        evaluation_predictions = []
        batch = int(config["batch_size"])
        with torch.no_grad():
            for start_index in range(0, len(eval_sequence), batch):
                seq_batch = eval_sequence[start_index : start_index + batch].to(device, non_blocking=True)
                cal_batch = eval_calendar[start_index : start_index + batch].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    prediction = model(seq_batch, cal_batch)
                evaluation_predictions.append(prediction.float().cpu().numpy())
        predicted_eval = np.vstack(evaluation_predictions) * y_std + y_mean
        aggregate_predictions.append(predicted_eval)
        values = point_metrics(actual_eval, predicted_eval)
        peak_gpu_memory = float(torch.cuda.max_memory_allocated() / 1024**3) if use_cuda else 0.0
        metric_rows.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "validation_mae_mw": best_loss,
                "runtime_seconds": time.perf_counter() - seed_started,
                "device": str(device),
                "peak_gpu_memory_gb": peak_gpu_memory,
                **values,
            }
        )
        prediction_frame: dict[str, Any] = {"target_date": evaluation_dates, "seed": seed}
        for hour_index in range(LOAD_HOURS):
            prediction_frame[TRUE_COLUMNS[hour_index]] = actual_eval[:, hour_index]
            prediction_frame[PRED_COLUMNS[hour_index]] = predicted_eval[:, hour_index]
        write_parquet(pd.DataFrame(prediction_frame), output_dir / f"tcn_seed_{seed}_predictions.parquet")
        log(f"TCN seed={seed}: evaluation MAE={values['mae_mw']:.3f} MW, best epoch={best_epoch}")
        del model
        if use_cuda:
            torch.cuda.empty_cache()

    mean_prediction = np.mean(np.stack(aggregate_predictions), axis=0)
    aggregate_metrics = point_metrics(actual_eval, mean_prediction)
    aggregate_frame: dict[str, Any] = {"target_date": evaluation_dates, "model": "TCN five-seed mean"}
    for hour_index in range(LOAD_HOURS):
        aggregate_frame[TRUE_COLUMNS[hour_index]] = actual_eval[:, hour_index]
        aggregate_frame[PRED_COLUMNS[hour_index]] = mean_prediction[:, hour_index]
    write_parquet(pd.DataFrame(aggregate_frame), output_dir / "tcn_five_seed_mean_predictions.parquet")
    write_csv(pd.DataFrame(metric_rows), output_dir / "tcn_seed_metrics.csv")
    write_csv(pd.DataFrame(history_rows), output_dir / "tcn_training_history.csv")
    architecture = {
        "input": "28 prior daily profiles as 24 channels by 28 days, plus 28 target-date calendar features",
        "convolution_channels": int(config["channels"]),
        "kernel_size": int(config["kernel_size"]),
        "dilations": list(config["dilations"]),
        "pooling": "adaptive average pooling over the 28-day sequence",
        "dense_units": int(config["dense_units"]),
        "dropout": float(config["dropout"]),
        "optimizer": "Adam",
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "batch_size": int(config["batch_size"]),
        "max_epochs": int(config["max_epochs"]),
        "early_stopping_patience": int(config["patience"]),
        "training_period": f"{train_start.date()} to {train_end.date()}",
        "validation_period": f"{validation_start.date()} to {validation_end.date()}",
        "evaluation_period": f"{evaluation_start.date()} to {evaluation_end.date()}",
        "device": str(device),
        "original_unidentified_path_status": "deprecated; all comparisons use named seed files or the explicit five-seed mean",
        "aggregate_metrics": aggregate_metrics,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(architecture, output_dir / "tcn_architecture_and_run.json")
    return {"status": "PASS", **architecture}
