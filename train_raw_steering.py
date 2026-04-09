#!/usr/bin/env python3
"""Fine-tune an existing raw-steering policy using old MPC datasets and new DAgger rollout logs.

This script is designed for the current project pipeline:
- old training datasets use columns like `prev_steering`, `expert_steering`
- new DAgger rollout logs use columns like `prev_steering_raw`, `raw_steering_expert`
- the previous model can be loaded either from `model_raw_steering.pt` or from
  `policy_export.json`

Main features:
1. Load one or more old expert CSV datasets.
2. Load one or more DAgger CSV logs and convert them to the training schema.
3. Aggregate all data into a single dataframe.
4. Fine-tune from an existing model with a small learning rate.
5. Optionally upweight DAgger / fallback samples.
6. Export an updated PyTorch checkpoint + JSON for C++ runtime.

Example (PowerShell one line):
    python finetune_with_dagger.py --base-csv dataset_1.csv dataset_2.csv dataset_3.csv dataset_4.csv \
        --dagger-csv policy_dagger_log.csv --resume policy_export.json --output-dir finetune_run
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

FEATURE_COLUMNS: List[str] = [
    "lateral_deviation",
    "yaw_angle",
    "curvature_0",
    "curvature_1",
    "curvature_2",
    "curvature_3",
    "velocity",
    "prev_steering",
]
TARGET_COLUMN = "expert_steering"


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], activation: str = "tanh") -> None:
        super().__init__()
        layers: List[nn.Module] = []
        dims = [in_dim, *hidden_dims, 1]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a raw-steering policy with DAgger data")
    parser.add_argument("--base-csv", nargs="*", default=[], help="Old expert CSV files from MPC-only collection")
    parser.add_argument("--dagger-csv", nargs="*", default=[], help="New DAgger rollout CSV files")
    parser.add_argument("--resume", required=True, help="Path to old model checkpoint (.pt) or policy_export.json")
    parser.add_argument("--output-dir", default="finetune_dagger_run", help="Directory to save outputs")
    parser.add_argument("--epochs", type=int, default=80, help="Max fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for fine-tuning")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Per-file train fraction")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Per-file validation fraction")
    parser.add_argument("--num-threads", type=int, default=1, help="Torch CPU threads")

    parser.add_argument("--max-abs-lateral", type=float, default=2.0)
    parser.add_argument("--max-abs-yaw", type=float, default=1.2)
    parser.add_argument("--max-abs-curvature", type=float, default=5.0)
    parser.add_argument("--max-abs-prev-steering", type=float, default=28.0)
    parser.add_argument("--max-abs-target", type=float, default=28.0)

    parser.add_argument("--dagger-weight", type=float, default=2.0, help="Sample weight for DAgger rows")
    parser.add_argument("--fallback-weight", type=float, default=3.0, help="Extra sample weight when fallback_to_mpc=1")
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Use weighted sampling instead of plain shuffle")
    parser.add_argument("--only-fallback", action="store_true", help="Train only on DAgger rows with fallback_to_mpc=1")
    parser.add_argument("--keep-policy-columns", action="store_true", help="Keep extra policy/debug columns in saved merged CSV")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_minimum_split_counts(n_rows: int, train_frac: float, val_frac: float) -> Tuple[int, int, int]:
    if n_rows < 3:
        raise ValueError(f"Need at least 3 rows per source file to create train/val/test splits; got {n_rows}")

    n_train = int(math.floor(n_rows * train_frac))
    n_val = int(math.floor(n_rows * val_frac))
    n_test = n_rows - n_train - n_val

    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1
    n_test = n_rows - n_train - n_val

    if n_test < 1:
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
        n_test = n_rows - n_train - n_val

    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"Could not create non-empty splits from {n_rows} rows")

    return n_train, n_val, n_test


def read_json_model(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def model_from_json(json_cfg: Dict[str, object]) -> Tuple[MLPRegressor, Dict[str, np.ndarray], Dict[str, object]]:
    hidden_dims = [int(v) for v in json_cfg["hidden_dims"]]
    activation = str(json_cfg["activation"])
    model = MLPRegressor(len(FEATURE_COLUMNS), hidden_dims, activation=activation)

    linear_layers: List[nn.Linear] = [m for m in model.modules() if isinstance(m, nn.Linear)]
    exported_layers = json_cfg["layers"]
    if len(linear_layers) != len(exported_layers):
        raise ValueError("JSON layer count does not match model architecture")

    with torch.no_grad():
        for layer_mod, layer_cfg in zip(linear_layers, exported_layers):
            w = torch.tensor(layer_cfg["weight"], dtype=torch.float32)
            b = torch.tensor(layer_cfg["bias"], dtype=torch.float32)
            if tuple(layer_mod.weight.shape) != tuple(w.shape):
                raise ValueError(f"Weight shape mismatch: model {tuple(layer_mod.weight.shape)} vs JSON {tuple(w.shape)}")
            if tuple(layer_mod.bias.shape) != tuple(b.shape):
                raise ValueError(f"Bias shape mismatch: model {tuple(layer_mod.bias.shape)} vs JSON {tuple(b.shape)}")
            layer_mod.weight.copy_(w)
            layer_mod.bias.copy_(b)

    norm_cfg = json_cfg["normalization"]
    norm = {
        "x_mean": np.asarray(norm_cfg["feature_mean"], dtype=np.float32),
        "x_std": np.asarray(norm_cfg["feature_std"], dtype=np.float32),
        "y_mean": np.asarray([norm_cfg["target_mean"]], dtype=np.float32),
        "y_std": np.asarray([norm_cfg["target_std"]], dtype=np.float32),
    }

    meta = {
        "hidden_dims": hidden_dims,
        "activation": activation,
        "feature_columns": list(json_cfg["feature_columns"]),
        "target_column": str(json_cfg.get("target", TARGET_COLUMN)),
        "source_format": "json",
    }
    return model, norm, meta


def load_resume_model(resume_path: str) -> Tuple[MLPRegressor, Dict[str, np.ndarray], Dict[str, object]]:
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume model not found: {path}")

    if path.suffix.lower() == ".json":
        cfg = read_json_model(path)
        return model_from_json(cfg)

    ckpt = torch.load(path, map_location="cpu")
    hidden_dims = [int(v) for v in ckpt["hidden_dims"]]
    activation = str(ckpt["activation"])
    model = MLPRegressor(len(FEATURE_COLUMNS), hidden_dims, activation=activation)
    model.load_state_dict(ckpt["state_dict"])

    raw_norm = ckpt["normalization"]
    norm = {
        "x_mean": np.asarray(raw_norm["x_mean"], dtype=np.float32),
        "x_std": np.asarray(raw_norm["x_std"], dtype=np.float32),
        "y_mean": np.asarray(raw_norm["y_mean"], dtype=np.float32).reshape(1),
        "y_std": np.asarray(raw_norm["y_std"], dtype=np.float32).reshape(1),
    }
    meta = {
        "hidden_dims": hidden_dims,
        "activation": activation,
        "feature_columns": list(ckpt.get("feature_columns", FEATURE_COLUMNS)),
        "target_column": str(ckpt.get("target_column", TARGET_COLUMN)),
        "source_format": "pt",
    }
    return model, norm, meta


def sanitize_base_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Base dataset {source_name} missing columns: {missing}")
    out = df.copy()
    out["dataset_role"] = "base"
    out["source_file"] = source_name
    if "fallback_to_mpc" not in out.columns:
        out["fallback_to_mpc"] = 0
    return out


def sanitize_dagger_df(df: pd.DataFrame, source_name: str, args: argparse.Namespace) -> pd.DataFrame:
    rename_map = {
        "prev_steering_raw": "prev_steering",
        "raw_steering_expert": "expert_steering",
    }
    out = df.rename(columns=rename_map).copy()

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"DAgger dataset {source_name} missing columns after rename: {missing}")

    if "is_valid" in out.columns:
        out = out[out["is_valid"] == 1].copy()
    if args.only_fallback:
        if "fallback_to_mpc" not in out.columns:
            raise ValueError("--only-fallback requested but fallback_to_mpc column is missing")
        out = out[out["fallback_to_mpc"] == 1].copy()

    out["dataset_role"] = "dagger"
    out["source_file"] = source_name
    if "fallback_to_mpc" not in out.columns:
        out["fallback_to_mpc"] = 0
    return out


def load_and_merge_datasets(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if not args.base_csv and not args.dagger_csv:
        raise ValueError("Provide at least one dataset via --base-csv and/or --dagger-csv")

    frames: List[pd.DataFrame] = []
    stats: Dict[str, object] = {"base": {}, "dagger": {}}

    for csv_path in args.base_csv:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Base CSV not found: {path}")
        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        stats["base"][path.name] = {"rows_before": int(len(df))}
        df = sanitize_base_df(df, path.name)
        stats["base"][path.name]["rows_after_schema"] = int(len(df))
        frames.append(df)

    for csv_path in args.dagger_csv:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"DAgger CSV not found: {path}")
        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        stats["dagger"][path.name] = {"rows_before": int(len(df))}
        df = sanitize_dagger_df(df, path.name, args)
        stats["dagger"][path.name]["rows_after_schema"] = int(len(df))
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    df_all = df_all.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    mask = np.ones(len(df_all), dtype=bool)
    mask &= np.abs(df_all["lateral_deviation"].to_numpy()) <= args.max_abs_lateral
    mask &= np.abs(df_all["yaw_angle"].to_numpy()) <= args.max_abs_yaw
    for col in ["curvature_0", "curvature_1", "curvature_2", "curvature_3"]:
        mask &= np.abs(df_all[col].to_numpy()) <= args.max_abs_curvature
    mask &= np.abs(df_all["prev_steering"].to_numpy()) <= args.max_abs_prev_steering
    mask &= np.abs(df_all[TARGET_COLUMN].to_numpy()) <= args.max_abs_target

    df_all = df_all.loc[mask].copy().reset_index(drop=True)

    if not args.keep_policy_columns:
        keep_cols = list(dict.fromkeys(FEATURE_COLUMNS + [TARGET_COLUMN, "dataset_role", "source_file", "fallback_to_mpc", "timestamp_ms", "frame_id"]))
        keep_cols = [c for c in keep_cols if c in df_all.columns]
        df_all = df_all[keep_cols].copy()

    summary = {
        "n_rows_total": int(len(df_all)),
        "n_rows_base": int((df_all["dataset_role"] == "base").sum()) if "dataset_role" in df_all.columns else 0,
        "n_rows_dagger": int((df_all["dataset_role"] == "dagger").sum()) if "dataset_role" in df_all.columns else 0,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "filters": {
            "max_abs_lateral": args.max_abs_lateral,
            "max_abs_yaw": args.max_abs_yaw,
            "max_abs_curvature": args.max_abs_curvature,
            "max_abs_prev_steering": args.max_abs_prev_steering,
            "max_abs_target": args.max_abs_target,
        },
        "input_stats": stats,
    }
    return df_all, summary


def split_per_source(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    sort_columns = [col for col in ["timestamp_ms", "frame_id"] if col in df.columns]

    for src_name, group in df.groupby("source_file", sort=False):
        if sort_columns:
            group = group.sort_values(sort_columns)
        group = group.reset_index(drop=True)
        n_train, n_val, _ = _ensure_minimum_split_counts(len(group), train_frac, val_frac)
        train_parts.append(group.iloc[:n_train].copy())
        val_parts.append(group.iloc[n_train:n_train + n_val].copy())
        test_parts.append(group.iloc[n_train + n_val:].copy())

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, val_df, test_df


def dataframe_to_tensors(df: pd.DataFrame, norm: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)

    x_norm = (x - norm["x_mean"]) / norm["x_std"]
    y_norm = (y - norm["y_mean"]) / norm["y_std"]

    x_tensor = torch.from_numpy(x_norm.astype(np.float32))
    y_tensor = torch.from_numpy(y_norm.astype(np.float32))
    return x_tensor, y_tensor, x, y


@torch.no_grad()
def predict_raw(model: nn.Module, x_tensor: torch.Tensor, norm: Dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    preds_norm = model(x_tensor.to(device)).cpu().numpy()
    preds = preds_norm * norm["y_std"] + norm["y_mean"]
    return preds.reshape(-1)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_abs_err = float(np.max(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if denom < 1e-12 else float(1.0 - np.sum(err ** 2) / denom)
    return {"mae": mae, "rmse": rmse, "max_abs_err": max_abs_err, "r2": r2}


def make_train_loader(train_df: pd.DataFrame, x_t: torch.Tensor, y_t: torch.Tensor, args: argparse.Namespace) -> DataLoader:
    ds = TensorDataset(x_t, y_t)

    if not args.use_weighted_sampler:
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    weights = np.ones(len(train_df), dtype=np.float32)
    if "dataset_role" in train_df.columns:
        weights = np.where(train_df["dataset_role"].to_numpy() == "dagger", args.dagger_weight, 1.0).astype(np.float32)
    if "fallback_to_mpc" in train_df.columns:
        fb = train_df["fallback_to_mpc"].to_numpy().astype(np.float32)
        weights *= np.where(fb > 0.5, args.fallback_weight, 1.0).astype(np.float32)

    sampler = WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
    return DataLoader(ds, batch_size=args.batch_size, sampler=sampler, drop_last=False)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_x: torch.Tensor,
    val_y_raw: np.ndarray,
    norm: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    history: List[Dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_pred = predict_raw(model, val_x, norm, device)
        val_metrics = regression_metrics(val_y_raw.reshape(-1), val_pred)

        row = {
            "epoch": epoch,
            "train_loss_norm_mse": train_loss,
            "val_rmse_raw": val_metrics["rmse"],
            "val_mae_raw": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
        }
        history.append(row)

        if val_metrics["rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = val_metrics["rmse"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:4d} | train MSE(norm)={train_loss:.6f} | "
                f"val RMSE(raw)={val_metrics['rmse']:.4f} | "
                f"val MAE(raw)={val_metrics['mae']:.4f} | val R2={val_metrics['r2']:.4f}"
            )

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best val RMSE={best_val_rmse:.4f})")
            break

    model.load_state_dict(best_state)
    return model, history


def export_model_to_json(
    model: nn.Module,
    output_path: Path,
    norm: Dict[str, np.ndarray],
    hidden_dims: Sequence[int],
    activation: str,
    training_summary: Dict[str, object],
) -> None:
    linear_layers: List[nn.Linear] = [m for m in model.modules() if isinstance(m, nn.Linear)]
    export = {
        "format_version": 1,
        "model_type": "mlp_regressor",
        "target": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "hidden_dims": list(hidden_dims),
        "activation": activation,
        "normalization": {
            "feature_mean": norm["x_mean"].astype(float).tolist(),
            "feature_std": norm["x_std"].astype(float).tolist(),
            "target_mean": float(norm["y_mean"][0]),
            "target_std": float(norm["y_std"][0]),
        },
        "layers": [
            {
                "weight": layer.weight.detach().cpu().numpy().astype(float).tolist(),
                "bias": layer.bias.detach().cpu().numpy().astype(float).tolist(),
            }
            for layer in linear_layers
        ],
        "training_summary": training_summary,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)


def save_plots(output_dir: Path, history: List[Dict[str, float]], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    hist_df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss_norm_mse"], label="train loss (norm MSE)")
    plt.plot(hist_df["epoch"], hist_df["val_rmse_raw"], label="val RMSE (raw steering)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / RMSE")
    plt.title("Fine-tuning history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true.reshape(-1), y_pred.reshape(-1), s=8, alpha=0.5)
    lim_min = float(min(np.min(y_true), np.min(y_pred)))
    lim_max = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lim_min, lim_max], [lim_min, lim_max])
    plt.xlabel("Expert steering")
    plt.ylabel("Predicted steering")
    plt.title("Test predictions vs expert")
    plt.tight_layout()
    plt.savefig(output_dir / "test_scatter.png", dpi=150)
    plt.close()

    n_show = min(400, len(y_true))
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:n_show].reshape(-1), label="expert")
    plt.plot(y_pred[:n_show].reshape(-1), label="predicted")
    plt.xlabel("Sample index")
    plt.ylabel("Raw steering")
    plt.title(f"First {n_show} test samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_timeseries.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.set_num_threads(max(1, int(args.num_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = pick_device(args.device)

    model, old_norm, resume_meta = load_resume_model(args.resume)
    if resume_meta["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError(
            f"Resume model feature order mismatch. Got {resume_meta['feature_columns']}, expected {FEATURE_COLUMNS}"
        )
    if resume_meta["target_column"] != TARGET_COLUMN:
        raise ValueError(
            f"Resume model target mismatch. Got {resume_meta['target_column']}, expected {TARGET_COLUMN}"
        )

    df_all, dataset_summary = load_and_merge_datasets(args)
    if len(df_all) < 30:
        raise ValueError(f"Too few rows after filtering: {len(df_all)}")

    train_df, val_df, test_df = split_per_source(df_all, args.train_frac, args.val_frac)

    # Recompute normalization from aggregated TRAIN split.
    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)
    norm = {
        "x_mean": x_train.mean(axis=0),
        "x_std": np.where(x_train.std(axis=0) < 1e-8, 1.0, x_train.std(axis=0)),
        "y_mean": y_train.mean(axis=0),
        "y_std": np.where(y_train.std(axis=0) < 1e-8, 1.0, y_train.std(axis=0)),
    }

    train_x_t, train_y_t, _, train_y_raw = dataframe_to_tensors(train_df, norm)
    val_x_t, val_y_t, _, val_y_raw = dataframe_to_tensors(val_df, norm)
    test_x_t, test_y_t, _, test_y_raw = dataframe_to_tensors(test_df, norm)

    train_loader = make_train_loader(train_df, train_x_t, train_y_t, args)
    model = model.to(device)

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_x=val_x_t,
        val_y_raw=val_y_raw,
        norm=norm,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    train_pred = predict_raw(model, train_x_t, norm, device)
    val_pred = predict_raw(model, val_x_t, norm, device)
    test_pred = predict_raw(model, test_x_t, norm, device)

    metrics = {
        "train": regression_metrics(train_y_raw.reshape(-1), train_pred),
        "val": regression_metrics(val_y_raw.reshape(-1), val_pred),
        "test": regression_metrics(test_y_raw.reshape(-1), test_pred),
        "n_rows": {
            "total_after_filter": int(len(df_all)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "resume_model": args.resume,
        "resume_format": resume_meta["source_format"],
        "device": str(device),
        "hidden_dims": resume_meta["hidden_dims"],
        "activation": resume_meta["activation"],
        "seed": args.seed,
        "fine_tune": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dagger_weight": args.dagger_weight,
            "fallback_weight": args.fallback_weight,
            "use_weighted_sampler": bool(args.use_weighted_sampler),
            "only_fallback": bool(args.only_fallback),
        },
        "old_normalization": {
            "x_mean": old_norm["x_mean"].astype(float).tolist(),
            "x_std": old_norm["x_std"].astype(float).tolist(),
            "y_mean": old_norm["y_mean"].astype(float).tolist(),
            "y_std": old_norm["y_std"].astype(float).tolist(),
        },
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "hidden_dims": resume_meta["hidden_dims"],
            "activation": resume_meta["activation"],
            "normalization": {
                "x_mean": norm["x_mean"],
                "x_std": norm["x_std"],
                "y_mean": norm["y_mean"],
                "y_std": norm["y_std"],
            },
        },
        output_dir / "model_raw_steering_finetuned.pt",
    )

    export_model_to_json(
        model=model,
        output_path=output_dir / "policy_export_finetuned.json",
        norm=norm,
        hidden_dims=resume_meta["hidden_dims"],
        activation=resume_meta["activation"],
        training_summary={"metrics": metrics, "dataset_summary": dataset_summary},
    )

    df_all.to_csv(output_dir / "merged_training_dataset.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame({"y_true": test_y_raw.reshape(-1), "y_pred": test_pred.reshape(-1)}).to_csv(
        output_dir / "test_predictions.csv", index=False
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    if not args.no_plots:
        save_plots(output_dir, history, test_y_raw, test_pred)

    print("\n=== Final metrics ===")
    for split_name in ["train", "val", "test"]:
        split_metrics = metrics[split_name]
        print(
            f"{split_name:>5s}: RMSE={split_metrics['rmse']:.4f}, "
            f"MAE={split_metrics['mae']:.4f}, R2={split_metrics['r2']:.4f}, "
            f"MaxAbsErr={split_metrics['max_abs_err']:.4f}"
        )

    print("\nSaved outputs to:", output_dir)
    print("- merged_training_dataset.csv")
    print("- model_raw_steering_finetuned.pt")
    print("- policy_export_finetuned.json")
    print("- metrics.json")
    print("- dataset_summary.json")
    print("- training_history.csv")
    print("- test_predictions.csv")
    if not args.no_plots:
        print("- training_history.png")
        print("- test_scatter.png")
        print("- test_timeseries.png")


if __name__ == "__main__":
    main()
