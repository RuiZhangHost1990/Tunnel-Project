from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from .config import NTUConfig


class TimeSeriesDataset(Dataset):
    def __init__(self, features, time_steps, history_values, future_values, model_ids=None):
        self.features = torch.FloatTensor(features)
        self.time_steps = torch.FloatTensor(time_steps)
        self.history_values = torch.FloatTensor(history_values)
        self.future_values = torch.FloatTensor(future_values)
        self.model_ids = list(model_ids) if model_ids is not None else [None] * len(self.features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.time_steps[idx],
            self.history_values[idx],
            self.future_values[idx],
            self.model_ids[idx],
        )


class SimpleRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, num_points, future_steps):
        super().__init__()
        self.num_points = num_points
        self.future_steps = future_steps
        self.rnn = nn.GRU(
            input_size=num_points,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.feature_proj = nn.Linear(20, 16)
        self.time_proj = nn.Linear(1, 8)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 16 + 8, 128),
            nn.ReLU(),
            nn.Linear(128, future_steps * num_points),
        )

    def forward(self, features, time_steps, history):
        batch_size = history.size(0)
        rnn_out, _ = self.rnn(history)
        last_hidden = rnn_out[:, -1, :]
        features_proj = self.feature_proj(features)
        time_proj = self.time_proj(time_steps)
        combined = torch.cat([last_hidden, features_proj, time_proj], dim=1)
        output = self.fc(combined)
        return output.view(batch_size, self.future_steps, self.num_points)


class CombinedModel(nn.Module):
    def __init__(self, num_points=30, future_steps=5):
        super().__init__()
        self.rnn_model = SimpleRNN(
            hidden_size=128,
            num_layers=2,
            num_points=num_points,
            future_steps=future_steps,
        )

    def forward(self, features, time_steps, history):
        return self.rnn_model(features, time_steps, history)


def build_stage1_model(cfg: NTUConfig, model_type=None):
    """Build a model explicitly; old checkpoints without a type remain GRU."""
    kind = (model_type or cfg.stage1_model_type).lower()
    if kind == "gru":
        return CombinedModel(cfg.stage1_num_points, cfg.stage1_future_steps)
    if kind == "cnn_patchtst":
        from .stage1_cnn_patchtst import Stage1CNNPatchTST
        return Stage1CNNPatchTST(cfg.stage1_num_points, cfg.stage1_history_steps,
                                 cfg.stage1_future_steps)
    raise ValueError(f"Unknown Stage-I model type: {kind!r}")


def stage1_split_indices(cfg: NTUConfig, n_samples: int):
    """Split whole FE cases, preserving file order unless shuffling is requested."""
    if n_samples < 2:
        raise ValueError("Stage-I needs at least two cases for train/validation splitting.")
    if cfg.stage1_train_count is None:
        if not 0.0 < cfg.stage1_train_ratio < 1.0:
            raise ValueError("stage1_train_ratio must be between 0 and 1.")
        n_train = max(1, int(n_samples * cfg.stage1_train_ratio))
    else:
        n_train = cfg.stage1_train_count
    if isinstance(n_train, bool) or not isinstance(n_train, (int, np.integer)):
        raise ValueError("stage1_train_count must be an integer.")
    if not 1 <= n_train < n_samples:
        raise ValueError("Training count must leave at least one validation case.")
    indices = np.arange(n_samples)
    if cfg.stage1_shuffle_split:
        indices = np.random.default_rng(cfg.random_state).permutation(indices)
    return indices[:n_train], indices[n_train:]


def _validate_stage1_indices(train_indices, validation_indices, n_samples):
    groups = []
    for name, values in (("train", train_indices), ("validation", validation_indices)):
        idx = np.asarray(values)
        if idx.ndim != 1 or len(idx) == 0 or idx.dtype.kind not in "iu":
            raise ValueError(f"{name} indices must be a nonempty integer vector.")
        if len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= n_samples):
            raise ValueError(f"Invalid or duplicate {name} case indices.")
        groups.append(idx.astype(int))
    if np.intersect1d(*groups).size:
        raise ValueError("Training and validation cases must be disjoint.")
    return tuple(groups)


def preprocess_stage1_data(cfg: NTUConfig, train_indices=None, validation_indices=None):
    header = 0 if cfg.stage1_has_header else None
    df = pd.read_excel(cfg.stage1_excel_path, engine="openpyxl", header=header).dropna()
    history_steps = cfg.stage1_history_steps
    future_steps = cfg.stage1_future_steps
    total_steps = history_steps + future_steps
    num_points = cfg.stage1_num_points

    model_ids = []
    processed_features = []
    processed_time_steps = []
    processed_history = []
    processed_future = []

    for model_id in df.iloc[:, 0].unique():
        model_data = df[df.iloc[:, 0] == model_id]
        if len(model_data) < total_steps:
            continue
        model_ids.append(model_id)
        processed_features.append(model_data.iloc[0, 1:21].values)
        processed_time_steps.append(model_data.iloc[0, 21])
        model_values = model_data.iloc[:, 22 : 22 + num_points].values
        processed_history.append(model_values[:history_steps, :])
        processed_future.append(model_values[history_steps : history_steps + future_steps, :])

    if not processed_features:
        raise ValueError(
            "Stage-I data has no usable model groups. Check history/future steps "
            f"({history_steps}+{future_steps}) and the first-column model IDs."
        )

    processed_features = np.asarray(processed_features, dtype=np.float32)
    raw_features = processed_features.copy()
    processed_time_steps = np.asarray(processed_time_steps, dtype=np.float32).reshape(-1, 1)
    processed_history = np.asarray(processed_history, dtype=np.float32)
    processed_future = np.asarray(processed_future, dtype=np.float32)

    feature_scaler = MinMaxScaler()
    time_scaler = MinMaxScaler()
    value_scaler = MinMaxScaler()

    n_samples = len(processed_features)
    if train_indices is None and validation_indices is None:
        train_indices, validation_indices = stage1_split_indices(cfg, n_samples)
    elif train_indices is None or validation_indices is None:
        raise ValueError("Provide both training and validation indices.")
    train_indices, validation_indices = _validate_stage1_indices(
        train_indices, validation_indices, n_samples
    )
    # Fit every scaler on training cases only, then transform held-out cases.
    feature_scaler.fit(processed_features[train_indices])
    time_scaler.fit(processed_time_steps[train_indices])
    value_scaler.fit(np.vstack([
        processed_history[train_indices].reshape(-1, num_points),
        processed_future[train_indices].reshape(-1, num_points),
    ]))
    processed_features = feature_scaler.transform(processed_features)
    processed_time_steps = time_scaler.transform(processed_time_steps)
    history_normalized = value_scaler.transform(
        processed_history.reshape(-1, num_points)
    ).reshape(processed_history.shape)
    future_normalized = value_scaler.transform(
        processed_future.reshape(-1, num_points)
    ).reshape(processed_future.shape)

    return {
        "model_ids": np.asarray(model_ids, dtype=object),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "raw_features": raw_features,
        "features": processed_features,
        "time_steps": processed_time_steps,
        "history": history_normalized,
        "future": future_normalized,
        "feature_scaler": feature_scaler,
        "time_scaler": time_scaler,
        "value_scaler": value_scaler,
    }


def _make_loader(data, indices, cfg: NTUConfig, shuffle=False, include_ids=True):
    model_ids = data["model_ids"][indices] if include_ids else None
    dataset = TimeSeriesDataset(
        data["features"][indices],
        data["time_steps"][indices],
        data["history"][indices],
        data["future"][indices],
        model_ids=model_ids,
    )
    return DataLoader(dataset, batch_size=cfg.stage1_batch_size, shuffle=shuffle)


def train_stage1_model(cfg: NTUConfig, device=None, *, train_indices=None,
                       validation_indices=None, evaluation_indices=None, save_excel=True,
                       checkpoint_selection="best_validation"):
    out = cfg.resolved_output_dir()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    if checkpoint_selection not in ("best_validation", "last_epoch"):
        raise ValueError("checkpoint_selection must be best_validation or last_epoch.")
    if cfg.stage1_epochs < 1 or cfg.stage1_batch_size < 1:
        raise ValueError("Stage-I epochs and batch size must be positive.")
    data = preprocess_stage1_data(cfg, train_indices, validation_indices)
    n_samples = len(data["features"])
    train_idx = data["train_indices"]
    test_idx = data["validation_indices"]
    evaluation_idx = np.empty(0, dtype=int)
    if evaluation_indices is not None:
        _, evaluation_idx = _validate_stage1_indices(train_idx, evaluation_indices, n_samples)
        if np.intersect1d(test_idx, evaluation_idx).size:
            raise ValueError("Validation and evaluation cases must be disjoint.")
    split_rows = [{"model_id": model_id,
                   "split": "train" if i in train_idx else "validation" if i in test_idx else "test" if i in evaluation_idx else "unused"}
                  for i, model_id in enumerate(data["model_ids"])]
    pd.DataFrame(split_rows).to_csv(out / "tables" / "stage1_split.csv", index=False)
    print(f"Stage-I cases: train={len(train_idx)}, validation={len(test_idx)}, test={len(evaluation_idx)}, total={n_samples}")

    train_loader = _make_loader(data, train_idx, cfg, shuffle=True)
    val_loader = _make_loader(data, test_idx, cfg, shuffle=False)
    all_loader = _make_loader(data, np.arange(n_samples), cfg, shuffle=False)

    model = build_stage1_model(cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.stage1_learning_rate)
    best_path = out / "models" / cfg.stage1_checkpoint_name
    history = []
    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(1, cfg.stage1_epochs + 1):
        model.train()
        train_loss = 0.0
        for features, time_steps, hist, future, _model_ids in train_loader:
            features = features.to(device)
            time_steps = time_steps.to(device)
            hist = hist.to(device)
            future = future.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features, time_steps, hist)
            loss = criterion(outputs, future)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(features)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, time_steps, hist, future, _model_ids in val_loader:
                features = features.to(device)
                time_steps = time_steps.to(device)
                hist = hist.to(device)
                future = future.to(device)
                val_loss += criterion(model(features, time_steps, hist), future).item() * len(features)

        train_loss /= len(train_idx)
        val_loss /= len(test_idx)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if ((checkpoint_selection == "best_validation" and val_loss < best_val_loss)
                or (checkpoint_selection == "last_epoch" and epoch == cfg.stage1_epochs)):
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": cfg.stage1_model_type,
                    "feature_scaler": data["feature_scaler"],
                    "time_scaler": data["time_scaler"],
                    "value_scaler": data["value_scaler"],
                    "cfg": cfg.__dict__,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "checkpoint_selection": checkpoint_selection,
                    "train_model_ids": data["model_ids"][train_idx].tolist(),
                    "validation_model_ids": data["model_ids"][test_idx].tolist(),
                    "test_model_ids": data["model_ids"][evaluation_idx].tolist(),
                },
                best_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.stage1_epochs:
            print(f"[Stage-I {epoch:04d}/{cfg.stage1_epochs}] train={train_loss:.4e}, val={val_loss:.4e}")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    predictions, targets, model_ids = evaluate_stage1_model(model, all_loader, data["value_scaler"], device)
    pd.DataFrame(history).to_csv(out / "tables" / "stage1_training_history.csv", index=False)
    if save_excel:
        save_stage1_results(predictions, targets, model_ids, out / "tables" / "stage1_model_results.xlsx")
    metric_rows = []
    metric_groups = [("train", train_idx), ("validation", test_idx)]
    if len(evaluation_idx):
        metric_groups.append(("test", evaluation_idx))
    for split, idx in metric_groups:
        for step in [None, *range(cfg.stage1_future_steps)]:
            yt = targets[idx] if step is None else targets[idx, step]
            yp = predictions[idx] if step is None else predictions[idx, step]
            metric_rows.append({
                "split": split, "future_step": "all" if step is None else step + 1,
                "n_cases": len(idx), "n_values": yt.size,
                "MAE": float(np.mean(np.abs(yp - yt))),
                "RMSE": float(np.sqrt(np.mean((yp - yt) ** 2))),
                "R2": float(r2_score(yt.ravel(), yp.ravel())),
            })
    split_metrics = pd.DataFrame(metric_rows)
    split_metrics.to_csv(out / "tables" / "stage1_split_metrics.csv", index=False)

    return {
        "model": model,
        "predictions": predictions,
        "targets": targets,
        "model_ids": model_ids,
        "raw_features": data["raw_features"],
        "best_model_path": best_path,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "checkpoint_selection": checkpoint_selection,
        "split_metrics": split_metrics,
        "train_indices": train_idx,
        "validation_indices": test_idx,
        "evaluation_indices": evaluation_idx,
    }


def load_stage1_checkpoint(cfg: NTUConfig, checkpoint_path=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else cfg.resolved_output_dir() / "models" / cfg.stage1_checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Reconstruct dimensions/type from the saved model, independent of caller defaults.
    from dataclasses import fields
    allowed = {f.name for f in fields(NTUConfig)}
    saved_cfg = NTUConfig(**{k: v for k, v in checkpoint.get("cfg", {}).items() if k in allowed})
    kind = checkpoint.get("model_type", "gru").lower()
    model = build_stage1_model(saved_cfg, model_type=kind).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return {
        "model": model,
        "model_type": kind,
        "cfg": saved_cfg,
        "feature_scaler": checkpoint["feature_scaler"],
        "time_scaler": checkpoint["time_scaler"],
        "value_scaler": checkpoint["value_scaler"],
        "best_val_loss": checkpoint.get("best_val_loss"),
        "checkpoint_path": checkpoint_path,
    }


def predict_stage1_from_excel(cfg: NTUConfig, checkpoint=None, device=None):
    """Forecast from observed history; future rows are optional evaluation data."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint is None:
        checkpoint = load_stage1_checkpoint(cfg, device=device)
    model = checkpoint["model"]
    saved_cfg = checkpoint.get("cfg", cfg)
    for name in ("stage1_history_steps", "stage1_future_steps", "stage1_num_points"):
        if getattr(saved_cfg, name) != getattr(cfg, name):
            raise ValueError(f"Inference {name} does not match the trained checkpoint.")
    header = 0 if cfg.stage1_has_header else None
    df = pd.read_excel(cfg.stage1_excel_path, engine="openpyxl", header=header).dropna()
    history_steps = cfg.stage1_history_steps
    future_steps = cfg.stage1_future_steps
    total_steps = history_steps + future_steps
    num_points = cfg.stage1_num_points

    model_ids = []
    raw_features = []
    features = []
    time_steps = []
    history = []
    future = []
    for model_id in df.iloc[:, 0].unique():
        model_data = df[df.iloc[:, 0] == model_id]
        if len(model_data) < history_steps:
            continue
        model_ids.append(model_id)
        feature_row = model_data.iloc[0, 1:21].values
        raw_features.append(feature_row)
        features.append(feature_row)
        time_steps.append(model_data.iloc[0, 21])
        model_values = model_data.iloc[:, 22 : 22 + num_points].values.astype(np.float32)
        history.append(model_values[:history_steps, :])
        if len(model_values) >= total_steps:
            future.append(model_values[history_steps : total_steps, :])
        else:
            future.append(np.full((future_steps, num_points), np.nan, dtype=np.float32))

    if not model_ids:
        raise ValueError(f"No cases have the required {history_steps} observed steps.")
    features = checkpoint["feature_scaler"].transform(np.asarray(features, dtype=np.float32))
    time_steps = checkpoint["time_scaler"].transform(np.asarray(time_steps, dtype=np.float32).reshape(-1, 1))
    history = np.asarray(history, dtype=np.float32)
    future = np.asarray(future, dtype=np.float32)
    flat_history = history.reshape(-1, num_points)
    history_norm = checkpoint["value_scaler"].transform(flat_history).reshape(history.shape)

    with torch.no_grad():
        pred_norm = model(
            torch.tensor(features, dtype=torch.float32, device=device),
            torch.tensor(time_steps, dtype=torch.float32, device=device),
            torch.tensor(history_norm, dtype=torch.float32, device=device),
        ).cpu().numpy()

    pred_2d = pred_norm.reshape(-1, num_points)
    predictions = checkpoint["value_scaler"].inverse_transform(pred_2d).reshape(
        len(model_ids), future_steps, num_points
    )
    return {
        "predictions": predictions,
        "targets": future,
        "model_ids": np.asarray(model_ids, dtype=object),
        "raw_features": np.asarray(raw_features, dtype=np.float32),
    }


def evaluate_stage1_model(model, loader, value_scaler, device):
    model.eval()
    all_predictions = []
    all_targets = []
    all_model_ids = []
    with torch.no_grad():
        for features, time_steps, hist, future, model_ids in loader:
            features = features.to(device)
            time_steps = time_steps.to(device)
            hist = hist.to(device)
            outputs = model(features, time_steps, hist)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(future.numpy())
            all_model_ids.extend(list(model_ids))

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    pred_2d = predictions.reshape(-1, predictions.shape[-1])
    target_2d = targets.reshape(-1, targets.shape[-1])
    predictions_original = value_scaler.inverse_transform(pred_2d).reshape(predictions.shape)
    targets_original = value_scaler.inverse_transform(target_2d).reshape(targets.shape)
    return predictions_original, targets_original, np.asarray(all_model_ids, dtype=object)


def save_stage1_results(predictions, targets, model_ids, output_path):
    rows = []
    for i, model_id in enumerate(model_ids):
        for t in range(predictions.shape[1]):
            for p in range(predictions.shape[2]):
                true = targets[i, t, p]
                pred = predictions[i, t, p]
                rows.append(
                    {
                        "model_id": model_id,
                        "future_step": t + 1,
                        "monitor_point": p + 1,
                        "true_value": true,
                        "predicted_value": pred,
                        "absolute_error": abs(pred - true),
                    }
                )
    detail = pd.DataFrame(rows)
    stats = (
        detail.groupby(["future_step", "monitor_point"])
        .apply(lambda g: pd.Series({"MAE": np.mean(g["absolute_error"]), "RMSE": np.sqrt(np.mean(g["absolute_error"] ** 2))}))
        .reset_index()
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        detail.to_excel(writer, sheet_name="Detailed Results", index=False)
        stats.to_excel(writer, sheet_name="Statistics", index=False)


def _selected_monitor_indices(num_points: int, keep: int, strategy: str):
    keep = int(min(max(keep, 1), num_points))
    if strategy == "first":
        return np.arange(keep)
    if strategy == "center":
        order = np.argsort(np.abs(np.arange(num_points) - (num_points - 1) / 2.0))
        return np.sort(order[:keep])
    return np.unique(np.round(np.linspace(0, num_points - 1, keep)).astype(int))


def flatten_stage1_predictions_for_stage2(cfg: NTUConfig, predictions):
    selected = _selected_monitor_indices(
        cfg.stage1_num_points,
        cfg.stage1_points_per_future_step,
        cfg.stage1_point_strategy,
    )
    stage2_inputs = predictions[:, :, selected].reshape(predictions.shape[0], -1)
    expected_cols = cfg.settlement_feature_count
    if stage2_inputs.shape[1] != expected_cols:
        raise ValueError(
            "Stage-I flattened prediction width does not match Stage-II settlement input count: "
            f"{stage2_inputs.shape[1]} != {expected_cols}. Adjust stage1_points_per_future_step."
        )
    return stage2_inputs, selected


def build_stage2_excel_with_stage1_predictions(cfg: NTUConfig, stage1_result):
    out = cfg.resolved_output_dir()
    df = pd.read_excel(cfg.excel_path, engine="openpyxl")
    data_start = 1 if cfg.skip_first_data_row else 0
    data_rows = np.arange(data_start, len(df))

    abs_start = cfg.input_start + cfg.settlement_feature_start_in_x
    abs_end = abs_start + cfg.settlement_feature_count
    predictions = stage1_result["predictions"]
    model_ids = [str(x) for x in stage1_result["model_ids"]]
    model_id_to_pred_index = {model_id: i for i, model_id in enumerate(model_ids)}
    raw_features = stage1_result.get("raw_features")
    feature_start = cfg.input_start
    feature_end = feature_start + 20

    used_mode = "row_order_flattened"
    replaced_count = 0
    feature_copied_count = 0
    selected = np.arange(cfg.stage1_num_points)

    if cfg.stage2_time_block_replacement:
        rows_per_step = int(cfg.stage2_rows_per_time_step)
        if rows_per_step <= 0:
            raise ValueError("stage2_rows_per_time_step must be a positive integer.")
        n_blocks = int(np.ceil(len(data_rows) / rows_per_step))
        n_future = predictions.shape[1]
        if n_blocks > n_future:
            raise ValueError(
                "Stage-II has more 22-row time blocks than Stage-I future steps: "
                f"{n_blocks} blocks > {n_future} future steps."
            )

        used_mode = "time_block_model_id"
        for block_id in range(n_blocks):
            block_rows = data_rows[block_id * rows_per_step : (block_id + 1) * rows_per_step]
            if len(block_rows) == 0:
                continue
            block_ids = df.iloc[block_rows, 0].astype(str).tolist() if df.shape[1] > 0 else []
            has_id_match = any(model_id in model_id_to_pred_index for model_id in block_ids)
            if not has_id_match:
                used_mode = "time_block_row_order"

            for local_i, row_i in enumerate(block_rows):
                model_id = block_ids[local_i] if local_i < len(block_ids) else None
                if has_id_match and model_id in model_id_to_pred_index:
                    pred_i = model_id_to_pred_index[model_id]
                else:
                    pred_i = local_i
                if pred_i >= predictions.shape[0]:
                    continue
                if raw_features is not None and pred_i < len(raw_features):
                    df.iloc[row_i, feature_start:feature_end] = raw_features[pred_i]
                    feature_copied_count += 1
                df.iloc[row_i, abs_start:abs_end] = predictions[pred_i, block_id, :]
                replaced_count += 1

    else:
        stage2_inputs, selected = flatten_stage1_predictions_for_stage2(cfg, predictions)
        replacement = pd.DataFrame(stage2_inputs, index=model_ids)
        used_mode = "row_order_flattened"
        if df.shape[1] > 0:
            stage2_ids = df.iloc[data_rows, 0].astype(str).tolist()
            if any(mid in replacement.index for mid in stage2_ids):
                used_mode = "model_id_flattened"
                for row_i, model_id in zip(data_rows, stage2_ids):
                    if model_id in replacement.index:
                        pred_i = model_id_to_pred_index[model_id]
                        if raw_features is not None and pred_i < len(raw_features):
                            df.iloc[row_i, feature_start:feature_end] = raw_features[pred_i]
                            feature_copied_count += 1
                        df.iloc[row_i, abs_start:abs_end] = replacement.loc[model_id].values
                        replaced_count += 1

        if replaced_count == 0:
            n = min(len(data_rows), len(stage2_inputs))
            if raw_features is not None:
                df.iloc[data_rows[:n], feature_start:feature_end] = raw_features[:n]
                feature_copied_count = n
            df.iloc[data_rows[:n], abs_start:abs_end] = stage2_inputs[:n]
            replaced_count = n

    out_excel = out / "tables" / cfg.stage1_replaced_excel_name
    df.to_excel(out_excel, index=False)

    mapping = pd.DataFrame(
        {
            "selected_monitor_point_index_zero_based": selected,
            "selected_monitor_point_one_based": selected + 1,
        }
    )
    mapping.to_csv(out / "tables" / "stage1_selected_monitor_points.csv", index=False)

    pd.DataFrame(
        {
            "replacement_mode": [used_mode],
            "replaced_rows": [replaced_count],
            "stage1_has_header": [cfg.stage1_has_header],
            "stage1_static_feature_rows_copied_to_B_U": [feature_copied_count],
            "stage1_static_feature_column_start_zero_based": [feature_start],
            "stage1_static_feature_column_end_zero_based_exclusive": [feature_end],
            "stage2_tunnel_columns_kept_from_original_start_zero_based": [feature_end],
            "stage2_tunnel_columns_kept_from_original_end_zero_based_exclusive": [abs_start],
            "stage2_rows_per_time_step": [cfg.stage2_rows_per_time_step],
            "stage2_time_blocks": [int(np.ceil(len(data_rows) / cfg.stage2_rows_per_time_step)) if cfg.stage2_rows_per_time_step else None],
            "stage2_settlement_column_start_zero_based": [abs_start],
            "stage2_settlement_column_end_zero_based_exclusive": [abs_end],
        }
    ).to_csv(out / "tables" / "stage1_to_stage2_replacement_summary.csv", index=False)

    return replace(cfg, excel_path=str(out_excel))


def prepare_stage2_config_with_stage1_predictions(cfg: NTUConfig, device=None):
    result = train_stage1_model(cfg, device=device)
    return build_stage2_excel_with_stage1_predictions(cfg, result), result
