#!/usr/bin/env python3
"""Train an imitation-learning policy from MPC expert datasets.

Expected CSV columns:
- lateral_deviation
- yaw_angle
- curvature_0
- curvature_1
- curvature_2
- curvature_3
- velocity
- prev_steering
- expert_steering

Optional columns:
- is_valid
- timestamp_ms
- frame_id

Outputs:
- model_raw_steering.pt          : PyTorch checkpoint
- policy_export.json             : JSON export for C++ runtime
- metrics.json                   : train/val/test metrics
- merged_dataset.csv             : merged filtered dataset
- training_history.csv           : per-epoch history
- training_loss.png              : train/val loss curves
- pred_vs_true_scatter.png       : test scatter plot
- pred_vs_true_timeseries.png    : test prediction preview

Example:
    python train_raw_steering.py \
        --csv dataset1.csv dataset2.csv dataset3.csv dataset4.csv dataset5.csv dataset6.csv \
        --output-dir run_bc
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    parser = argparse.ArgumentParser(description="Train imitation learning policy from MPC datasets")
    parser.add_argument("--csv", nargs="+", required=True, help="One or more CSV dataset files")
    parser.add_argument("--output-dir", default="train_run", help="Directory to save outputs")
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[32, 32], help="MLP hidden layer sizes")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--num-threads", type=int, default=1)

    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)

    parser.add_argument("--max-abs-lateral", type=float, default=2.0)
    parser.add_argument("--max-abs-yaw", type=float, default=1.2)
    parser.add_argument("--max-abs-curvature", type=float, default=5.0)
    parser.add_argument("--max-abs-prev-steering", type=float, default=28.0)
    parser.add_argument("--max-abs-target", type=float, default=28.0)

    parser.add_argument("--drop-invalid", action="store_true", help="Drop rows with is_valid != 1 if column exists")
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


def ensure_minimum_split_counts(n_rows: int, train_frac: float, val_frac: float) -> Tuple[int, int, int]:
    if n_rows < 3:
        raise ValueError(f"Need at least 3 rows per source file, got {n_rows}")

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
        raise ValueError(f"Could not create non-empty train/val/test splits from {n_rows} rows")

    return n_train, n_val, n_test


def load_and_filter_datasets(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, object]]:
    frames: List[pd.DataFrame] = []
    per_file_stats: Dict[str, Dict[str, int]] = {}

    for csv_path in args.csv:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
        per_file_stats[path.name] = {"rows_before": int(len(df))}

        missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset {path.name} missing required columns: {missing}")

        if args.drop_invalid and "is_valid" in df.columns:
            df = df[df["is_valid"] == 1].copy()

        df["source_file"] = path.name
        per_file_stats[path.name]["rows_after_schema"] = int(len(df))
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    df_all = df_all.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

    mask = np.ones(len(df_all), dtype=bool)
    mask &= np.abs(df_all["lateral_deviation"].to_numpy()) <= args.max_abs_lateral
    mask &= np.abs(df_all["yaw_angle"].to_numpy()) <= args.max_abs_yaw
    for col in ["curvature_0", "curvature_1", "curvature_2", "curvature_3"]:
        mask &= np.abs(df_all[col].to_numpy()) <= args.max_abs_curvature
    mask &= np.abs(df_all["prev_steering"].to_numpy()) <= args.max_abs_prev_steering
    mask &= np.abs(df_all[TARGET_COLUMN].to_numpy()) <= args.max_abs_target

    df_all = df_all.loc[mask].copy().reset_index(drop=True)

    extra_curvature_cols = [
        c for c in df_all.columns
        if c.startswith("curvature_") and c not in FEATURE_COLUMNS
    ]

    keep_cols = list(dict.fromkeys(
        FEATURE_COLUMNS
        + extra_curvature_cols
        + [TARGET_COLUMN, "source_file", "timestamp_ms", "frame_id", "is_valid"]
    ))
    keep_cols = [c for c in keep_cols if c in df_all.columns]
    df_all = df_all[keep_cols].copy()

    summary = {
        "n_rows_total": int(len(df_all)),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "input_files": args.csv,
        "per_file_stats": per_file_stats,
        "filters": {
            "max_abs_lateral": args.max_abs_lateral,
            "max_abs_yaw": args.max_abs_yaw,
            "max-abs-curvature": args.max_abs_curvature,
            "max_abs_prev_steering": args.max_abs_prev_steering,
            "max_abs_target": args.max_abs_target,
            "drop_invalid": bool(args.drop_invalid),
        },
    }
    return df_all, summary


def split_per_source(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    sort_columns = [c for c in ["timestamp_ms", "frame_id"] if c in df.columns]

    for src_name, group in df.groupby("source_file", sort=False):
        if sort_columns:
            group = group.sort_values(sort_columns)
        group = group.reset_index(drop=True)

        n_train, n_val, _ = ensure_minimum_split_counts(len(group), train_frac, val_frac)
        train_parts.append(group.iloc[:n_train].copy())
        val_parts.append(group.iloc[n_train:n_train + n_val].copy())
        test_parts.append(group.iloc[n_train + n_val:].copy())

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, val_df, test_df


def build_normalization(train_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)
    return {
        "x_mean": x_train.mean(axis=0),
        "x_std": np.where(x_train.std(axis=0) < 1e-8, 1.0, x_train.std(axis=0)),
        "y_mean": y_train.mean(axis=0),
        "y_std": np.where(y_train.std(axis=0) < 1e-8, 1.0, y_train.std(axis=0)),
    }


def dataframe_to_tensors(df: pd.DataFrame, norm: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)

    x_norm = (x - norm["x_mean"]) / norm["x_std"]
    y_norm = (y - norm["y_mean"]) / norm["y_std"]

    x_t = torch.from_numpy(x_norm.astype(np.float32))
    y_t = torch.from_numpy(y_norm.astype(np.float32))
    return x_t, y_t, x, y


@torch.no_grad()
def predict_raw(model: nn.Module, x_tensor: torch.Tensor, norm: Dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    pred_norm = model(x_tensor.to(device)).cpu().numpy()
    pred = pred_norm * norm["y_std"] + norm["y_mean"]
    return pred.reshape(-1)


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
            pred = model(xb)
            loss = criterion(pred, yb)
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
                f"val MAE(raw)={val_metrics['mae']:.4f} | val R2={val_metrics['r2']:.5f}"
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
    hist_df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss_norm_mse"], label="train loss (norm MSE)")
    plt.plot(hist_df["epoch"], hist_df["val_rmse_raw"], label="val RMSE (raw steering)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / RMSE")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=150)
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
    plt.savefig(output_dir / "pred_vs_true_scatter.png", dpi=150)
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
    plt.savefig(output_dir / "pred_vs_true_timeseries.png", dpi=150)
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

    df_all, dataset_summary = load_and_filter_datasets(args)
    if len(df_all) < 30:
        raise ValueError(f"Too few rows after filtering: {len(df_all)}")

    train_df, val_df, test_df = split_per_source(df_all, args.train_frac, args.val_frac)
    norm = build_normalization(train_df)

    train_x_t, train_y_t, _, train_y_raw = dataframe_to_tensors(train_df, norm)
    val_x_t, val_y_t, _, val_y_raw = dataframe_to_tensors(val_df, norm)
    test_x_t, test_y_t, _, test_y_raw = dataframe_to_tensors(test_df, norm)

    train_loader = DataLoader(
        TensorDataset(train_x_t, train_y_t),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = MLPRegressor(
        in_dim=len(FEATURE_COLUMNS),
        hidden_dims=args.hidden_dims,
        activation=args.activation,
    ).to(device)

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
        "model": {
            "hidden_dims": args.hidden_dims,
            "activation": args.activation,
        },
        "optimizer": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
        },
        "device": str(device),
        "seed": args.seed,
        "dataset_summary": dataset_summary,
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "hidden_dims": args.hidden_dims,
            "activation": args.activation,
            "normalization": {
                "x_mean": norm["x_mean"],
                "x_std": norm["x_std"],
                "y_mean": norm["y_mean"],
                "y_std": norm["y_std"],
            },
        },
        output_dir / "model_raw_steering.pt",
    )

    export_model_to_json(
        model=model,
        output_path=output_dir / "policy_export.json",
        norm=norm,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        training_summary=metrics,
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    df_all.to_csv(output_dir / "merged_dataset.csv", index=False)
    save_plots(output_dir, history, test_y_raw.reshape(-1), test_pred.reshape(-1))

    print("\n=== Final metrics ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
