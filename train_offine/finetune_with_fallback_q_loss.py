#!/usr/bin/env python3
"""Fine-tune a raw-steering policy from fallback / intervention datasets.

This script is designed for the current RL-for-MPC workflow:
- resume from an existing policy_export.json or .pt checkpoint
- read one or more expert/base CSV files and one or more fallback/DAgger CSV files
- optionally keep only intervention rows from the fallback logs
- train with a hybrid objective:
      L_total = alpha_bc * L_imitation
              + lambda_delta_bc * L_delta_imitation
              + beta_q * L_q_like_1step

The Q-like loss is a differentiable 1-step surrogate derived from the same linearized /
discretized bicycle model used by the C++ MPC controller. It is not the exact Q-loss from the
paper, but it captures task-level penalties on next-step lateral error, next-step yaw error,
steering magnitude, and steering smoothness.

Expected CSV support:
- base expert CSVs:
    lateral_deviation, yaw_angle, curvature_0..3, velocity, prev_steering, expert_steering
- fallback / DAgger CSVs (common variants supported automatically):
    prev_steering_raw -> prev_steering
    raw_steering_expert or raw_steering_mpc -> expert_steering
    raw_steering_policy or raw_steering_pred -> policy_steering (optional, for diagnostics)
    fallback_to_mpc / fallback / intervention / mpc_intervened -> fallback_to_mpc

Example:
    python finetune_with_fallback_q_loss.py \
        --dagger-csv policy_fallback_log.csv \
        --resume policy_export.json \
        --output-dir finetune_fallback_q_run \
        --use-weighted-sampler \
        --train-scope fallback_only \
        --alpha-bc 1.0 --lambda-delta-bc 0.2 --beta-q 1e-3
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

OPTIONAL_POLICY_COLUMNS = [
    "policy_steering",
    "raw_abs_error",
    "executed_steering",
    "timestamp_ms",
    "frame_id",
]


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
    parser = argparse.ArgumentParser(description="Fine-tune a raw-steering policy from fallback / DAgger data")
    parser.add_argument("--base-csv", nargs="*", default=[], help="Optional expert/base CSV files")
    parser.add_argument("--dagger-csv", nargs="*", default=[], help="Fallback / DAgger rollout CSV files")
    parser.add_argument("--resume", required=True, help="Path to old model checkpoint (.pt) or policy_export.json")
    parser.add_argument("--output-dir", default="finetune_fallback_q_run", help="Directory to save outputs")

    parser.add_argument("--epochs", type=int, default=100, help="Maximum fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Per-file train fraction")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Per-file validation fraction")
    parser.add_argument("--num-threads", type=int, default=1, help="Torch CPU threads")

    parser.add_argument("--train-scope", choices=["fallback_only", "all"], default="fallback_only",
                        help="For DAgger logs: keep only fallback/intervention rows, or all valid rows")
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Use weighted sampling instead of plain shuffle")
    parser.add_argument("--dagger-weight", type=float, default=2.0, help="Sample weight for DAgger rows")
    parser.add_argument("--fallback-weight", type=float, default=3.0, help="Extra sample weight when fallback_to_mpc=1")

    parser.add_argument("--alpha-bc", type=float, default=1.0, help="Weight for imitation MSE loss")
    parser.add_argument("--lambda-delta-bc", type=float, default=0.2, help="Weight for steering increment imitation loss")
    parser.add_argument("--beta-q", type=float, default=1e-3, help="Weight for 1-step Q-like loss")
    parser.add_argument("--disable-q", action="store_true", help="Disable Q-like loss and run imitation only")

    parser.add_argument("--max-abs-lateral", type=float, default=2.0)
    parser.add_argument("--max-abs-yaw", type=float, default=1.2)
    parser.add_argument("--max-abs-curvature", type=float, default=5.0)
    parser.add_argument("--max-abs-prev-steering", type=float, default=28.0)
    parser.add_argument("--max-abs-target", type=float, default=28.0)

    # MPC model parameters copied from current C++ controller defaults.
    parser.add_argument("--ts", type=float, default=0.071)
    parser.add_argument("--mass", type=float, default=2.3)
    parser.add_argument("--lf", type=float, default=0.132)
    parser.add_argument("--lr-veh", type=float, default=0.12)
    parser.add_argument("--caf", type=float, default=0.04)
    parser.add_argument("--car", type=float, default=0.02)
    parser.add_argument("--iz", type=float, default=0.04)

    parser.add_argument("--mpc-q1", type=float, default=1500.0, help="Weight on next-step lateral error")
    parser.add_argument("--mpc-q2", type=float, default=120.0, help="Weight on next-step yaw error")
    parser.add_argument("--mpc-r", type=float, default=5.0, help="Weight on steering magnitude")
    parser.add_argument("--mpc-s", type=float, default=2.0, help="Weight on steering increment")

    parser.add_argument("--keep-policy-columns", action="store_true", help="Keep debug/policy columns in merged CSV")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
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
        return json.load(f)


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
        return model_from_json(read_json_model(path))

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


def _coerce_binary(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    return (vals > 0.5).astype(np.int64)


def _rename_first_existing(df: pd.DataFrame, targets: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    for dst, candidates in targets.items():
        if dst in out.columns:
            continue
        for src in candidates:
            if src in out.columns:
                out = out.rename(columns={src: dst})
                break
    return out


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
    else:
        out["fallback_to_mpc"] = _coerce_binary(out["fallback_to_mpc"])
    return out


def sanitize_dagger_df(df: pd.DataFrame, source_name: str, args: argparse.Namespace) -> pd.DataFrame:
    out = _rename_first_existing(
        df,
        {
            "prev_steering": ["prev_steering_raw", "prev_raw_steering"],
            "expert_steering": ["raw_steering_expert", "raw_steering_mpc", "expert_raw_steering"],
            "policy_steering": ["raw_steering_policy", "raw_steering_pred", "u_policy", "policy_raw_steering"],
            "executed_steering": ["raw_steering_exec", "raw_steering_executed", "u_exec"],
            "fallback_to_mpc": ["fallback", "intervention", "mpc_intervened", "is_fallback"],
        },
    )

    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Fallback/DAgger dataset {source_name} missing columns after rename: {missing}")

    if "is_valid" in out.columns:
        out = out[out["is_valid"] == 1].copy()

    if "fallback_to_mpc" in out.columns:
        out["fallback_to_mpc"] = _coerce_binary(out["fallback_to_mpc"])
    else:
        out["fallback_to_mpc"] = 0

    if args.train_scope == "fallback_only":
        if "fallback_to_mpc" not in out.columns:
            raise ValueError("--train-scope fallback_only requested but fallback_to_mpc column is missing")
        out = out[out["fallback_to_mpc"] == 1].copy()

    out["dataset_role"] = "dagger"
    out["source_file"] = source_name
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
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
        stats["base"][path.name] = {"rows_before": int(len(df))}
        df = sanitize_base_df(df, path.name)
        stats["base"][path.name]["rows_after_schema"] = int(len(df))
        frames.append(df)

    for csv_path in args.dagger_csv:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"DAgger CSV not found: {path}")
        df = pd.read_csv(path, on_bad_lines="skip", engine="python")
        stats["dagger"][path.name] = {"rows_before": int(len(df))}
        df = sanitize_dagger_df(df, path.name, args)
        stats["dagger"][path.name]["rows_after_schema"] = int(len(df))
        if "fallback_to_mpc" in df.columns:
            stats["dagger"][path.name]["rows_fallback"] = int((df["fallback_to_mpc"] == 1).sum())
            stats["dagger"][path.name]["rows_non_fallback"] = int((df["fallback_to_mpc"] == 0).sum())
        frames.append(df)

    if not frames:
        raise ValueError("No rows loaded from the provided CSV files")

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    df_all = df_all.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    mask = np.ones(len(df_all), dtype=bool)
    mask &= np.abs(df_all["lateral_deviation"].to_numpy(dtype=np.float32)) <= args.max_abs_lateral
    mask &= np.abs(df_all["yaw_angle"].to_numpy(dtype=np.float32)) <= args.max_abs_yaw
    for col in ["curvature_0", "curvature_1", "curvature_2", "curvature_3"]:
        mask &= np.abs(df_all[col].to_numpy(dtype=np.float32)) <= args.max_abs_curvature
    mask &= np.abs(df_all["prev_steering"].to_numpy(dtype=np.float32)) <= args.max_abs_prev_steering
    mask &= np.abs(df_all[TARGET_COLUMN].to_numpy(dtype=np.float32)) <= args.max_abs_target
    df_all = df_all.loc[mask].copy().reset_index(drop=True)

    if not args.keep_policy_columns:
        keep_cols = list(dict.fromkeys(
            FEATURE_COLUMNS
            + [TARGET_COLUMN, "dataset_role", "source_file", "fallback_to_mpc"]
            + OPTIONAL_POLICY_COLUMNS
        ))
        keep_cols = [c for c in keep_cols if c in df_all.columns]
        df_all = df_all[keep_cols].copy()

    summary = {
        "n_rows_total": int(len(df_all)),
        "n_rows_base": int((df_all["dataset_role"] == "base").sum()) if "dataset_role" in df_all.columns else 0,
        "n_rows_dagger": int((df_all["dataset_role"] == "dagger").sum()) if "dataset_role" in df_all.columns else 0,
        "n_rows_fallback": int((df_all["fallback_to_mpc"] == 1).sum()) if "fallback_to_mpc" in df_all.columns else 0,
        "n_rows_non_fallback": int((df_all["fallback_to_mpc"] == 0).sum()) if "fallback_to_mpc" in df_all.columns else 0,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "train_scope": args.train_scope,
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
    x_raw = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_raw = df[TARGET_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)

    x_norm = (x_raw - norm["x_mean"]) / norm["x_std"]
    y_norm = (y_raw - norm["y_mean"]) / norm["y_std"]
    return torch.from_numpy(x_norm.astype(np.float32)), torch.from_numpy(y_norm.astype(np.float32)), x_raw, y_raw


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


def make_train_loader(
    train_df: pd.DataFrame,
    x_norm_t: torch.Tensor,
    y_norm_t: torch.Tensor,
    x_raw: np.ndarray,
    args: argparse.Namespace,
) -> DataLoader:
    x_raw_t = torch.from_numpy(x_raw.astype(np.float32))
    ds = TensorDataset(x_norm_t, y_norm_t, x_raw_t)

    if not args.use_weighted_sampler:
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    weights = np.ones(len(train_df), dtype=np.float32)
    if "dataset_role" in train_df.columns:
        weights *= np.where(train_df["dataset_role"].to_numpy() == "dagger", args.dagger_weight, 1.0).astype(np.float32)
    if "fallback_to_mpc" in train_df.columns:
        fb = train_df["fallback_to_mpc"].to_numpy(dtype=np.float32)
        weights *= np.where(fb > 0.5, args.fallback_weight, 1.0).astype(np.float32)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    return DataLoader(ds, batch_size=args.batch_size, sampler=sampler, drop_last=False)


def build_discrete_matrices_torch(vx: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = vx.device
    dtype = vx.dtype
    batch = vx.shape[0]
    vx = torch.clamp(vx, min=1e-3)

    m = float(args.mass)
    lf = float(args.lf)
    lr = float(args.lr_veh)
    caf = float(args.caf)
    car = float(args.car)
    iz = float(args.iz)
    ts = float(args.ts)

    A_c = torch.zeros((batch, 4, 4), dtype=dtype, device=device)
    B_c = torch.zeros((batch, 4, 2), dtype=dtype, device=device)

    A_c[:, 0, 1] = 1.0
    A_c[:, 1, 1] = -(2 * caf + 2 * car) / (m * vx)
    A_c[:, 1, 2] = (2 * caf + 2 * car) / m
    A_c[:, 1, 3] = (-2 * caf * lf + 2 * car * lr) / (m * vx)
    A_c[:, 2, 3] = 1.0
    A_c[:, 3, 1] = (-2 * caf * lf + 2 * car * lr) / (iz * vx)
    A_c[:, 3, 2] = (2 * caf * lf - 2 * car * lr) / iz
    A_c[:, 3, 3] = (-2 * caf * lf * lf - 2 * car * lr * lr) / (iz * vx)

    B_c[:, 1, 0] = 2 * caf / m
    B_c[:, 1, 1] = (-2 * caf * lf + 2 * car * lr) / (m * vx) - vx
    B_c[:, 3, 0] = 2 * caf * lf / iz
    B_c[:, 3, 1] = (-2 * caf * lf * lf - 2 * car * lr * lr) / (iz * vx)

    M = torch.zeros((batch, 6, 6), dtype=dtype, device=device)
    M[:, :4, :4] = A_c
    M[:, :4, 4:] = B_c
    M = M * ts

    Md = torch.matrix_exp(M)
    A_d = Md[:, :4, :4]
    B_d = Md[:, :4, 4:]
    B1_d = B_d[:, :, 0:1]
    B2_d = B_d[:, :, 1:2]
    return A_d, B1_d, B2_d


def compute_loss_terms(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    x_raw: torch.Tensor,
    norm: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    dtype = pred_norm.dtype
    device = pred_norm.device

    y_mean = torch.tensor(float(norm["y_mean"][0]), dtype=dtype, device=device)
    y_std = torch.tensor(float(norm["y_std"][0]), dtype=dtype, device=device)

    pred_deg = pred_norm.squeeze(1) * y_std + y_mean
    tgt_deg = target_norm.squeeze(1) * y_std + y_mean
    pred_rad = pred_deg * (math.pi / 180.0)
    tgt_rad = tgt_deg * (math.pi / 180.0)

    prev_deg = x_raw[:, 7]
    prev_rad = prev_deg * (math.pi / 180.0)

    loss_bc = torch.mean((pred_norm - target_norm) ** 2)
    loss_delta_bc = torch.mean(((pred_rad - prev_rad) - (tgt_rad - prev_rad)) ** 2)

    if args.disable_q or args.beta_q <= 0.0:
        loss_q = torch.zeros((), dtype=dtype, device=device)
        aux = {
            "q_mean": torch.zeros((), dtype=dtype, device=device),
            "ey1_sq_mean": torch.zeros((), dtype=dtype, device=device),
            "epsi1_sq_mean": torch.zeros((), dtype=dtype, device=device),
            "u_sq_mean": torch.zeros((), dtype=dtype, device=device),
            "du_sq_mean": torch.zeros((), dtype=dtype, device=device),
        }
        return loss_bc, {"loss_delta_bc": loss_delta_bc, "loss_q": loss_q, **aux}

    ey = x_raw[:, 0]
    epsi = x_raw[:, 1]
    curv0 = x_raw[:, 2]
    vx = x_raw[:, 6]

    x0 = torch.stack([ey, torch.zeros_like(ey), epsi, torch.zeros_like(ey)], dim=1)
    A_d, B1_d, B2_d = build_discrete_matrices_torch(vx, args)
    v0 = curv0 * vx

    x1 = torch.bmm(A_d, x0.unsqueeze(-1))
    x1 = x1 + B1_d * pred_rad.view(-1, 1, 1) + B2_d * v0.view(-1, 1, 1)
    x1 = x1.squeeze(-1)

    ey1 = x1[:, 0]
    epsi1 = x1[:, 2]
    du = pred_rad - prev_rad

    q_sample = (
        float(args.mpc_q1) * ey1.pow(2)
        + float(args.mpc_q2) * epsi1.pow(2)
        + float(args.mpc_r) * pred_rad.pow(2)
        + float(args.mpc_s) * du.pow(2)
    )
    loss_q = q_sample.mean()
    aux = {
        "loss_delta_bc": loss_delta_bc,
        "loss_q": loss_q,
        "q_mean": q_sample.mean(),
        "ey1_sq_mean": ey1.pow(2).mean(),
        "epsi1_sq_mean": epsi1.pow(2).mean(),
        "u_sq_mean": pred_rad.pow(2).mean(),
        "du_sq_mean": du.pow(2).mean(),
    }
    return loss_bc, aux


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
    args: argparse.Namespace,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[Dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_losses: List[float] = []
        bc_losses: List[float] = []
        delta_losses: List[float] = []
        q_losses: List[float] = []

        for xb, yb, xrawb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xrawb = xrawb.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)

            loss_bc, aux = compute_loss_terms(preds, yb, xrawb, norm, args)
            loss_delta = aux["loss_delta_bc"]
            loss_q = aux["loss_q"]
            loss = float(args.alpha_bc) * loss_bc + float(args.lambda_delta_bc) * loss_delta + float(args.beta_q) * loss_q
            loss.backward()
            optimizer.step()

            total_losses.append(float(loss.item()))
            bc_losses.append(float(loss_bc.item()))
            delta_losses.append(float(loss_delta.item()))
            q_losses.append(float(loss_q.item()))

        val_pred = predict_raw(model, val_x, norm, device)
        val_metrics = regression_metrics(val_y_raw.reshape(-1), val_pred)

        row = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(total_losses)) if total_losses else float("nan"),
            "train_bc_loss_norm_mse": float(np.mean(bc_losses)) if bc_losses else float("nan"),
            "train_delta_loss": float(np.mean(delta_losses)) if delta_losses else float("nan"),
            "train_q_loss": float(np.mean(q_losses)) if q_losses else float("nan"),
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
                f"Epoch {epoch:4d} | "
                f"train_total={row['train_total_loss']:.6f} | "
                f"train_bc={row['train_bc_loss_norm_mse']:.6f} | "
                f"train_delta={row['train_delta_loss']:.6f} | "
                f"train_q={row['train_q_loss']:.6f} | "
                f"val RMSE(raw)={row['val_rmse_raw']:.4f} | "
                f"val MAE(raw)={row['val_mae_raw']:.4f} | val R2={row['val_r2']:.4f}"
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
    plt.plot(hist_df["epoch"], hist_df["train_total_loss"], label="train total")
    plt.plot(hist_df["epoch"], hist_df["train_bc_loss_norm_mse"], label="train bc")
    plt.plot(hist_df["epoch"], hist_df["train_q_loss"], label="train q")
    plt.plot(hist_df["epoch"], hist_df["val_rmse_raw"], label="val RMSE raw")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / RMSE")
    plt.title("Fallback fine-tuning history")
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

    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)
    norm = {
        "x_mean": x_train.mean(axis=0),
        "x_std": np.where(x_train.std(axis=0) < 1e-8, 1.0, x_train.std(axis=0)),
        "y_mean": y_train.mean(axis=0),
        "y_std": np.where(y_train.std(axis=0) < 1e-8, 1.0, y_train.std(axis=0)),
    }

    train_x_t, train_y_t, train_x_raw, train_y_raw = dataframe_to_tensors(train_df, norm)
    val_x_t, val_y_t, val_x_raw, val_y_raw = dataframe_to_tensors(val_df, norm)
    test_x_t, test_y_t, test_x_raw, test_y_raw = dataframe_to_tensors(test_df, norm)

    train_loader = make_train_loader(train_df, train_x_t, train_y_t, train_x_raw, args)
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
        args=args,
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
            "train_fallback": int((train_df["fallback_to_mpc"] == 1).sum()) if "fallback_to_mpc" in train_df.columns else 0,
            "val_fallback": int((val_df["fallback_to_mpc"] == 1).sum()) if "fallback_to_mpc" in val_df.columns else 0,
            "test_fallback": int((test_df["fallback_to_mpc"] == 1).sum()) if "fallback_to_mpc" in test_df.columns else 0,
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
            "train_scope": args.train_scope,
            "dagger_weight": args.dagger_weight,
            "fallback_weight": args.fallback_weight,
            "use_weighted_sampler": bool(args.use_weighted_sampler),
            "alpha_bc": args.alpha_bc,
            "lambda_delta_bc": args.lambda_delta_bc,
            "beta_q": args.beta_q,
            "disable_q": bool(args.disable_q),
            "mpc_q1": args.mpc_q1,
            "mpc_q2": args.mpc_q2,
            "mpc_r": args.mpc_r,
            "mpc_s": args.mpc_s,
            "ts": args.ts,
            "mass": args.mass,
            "lf": args.lf,
            "lr_veh": args.lr_veh,
            "caf": args.caf,
            "car": args.car,
            "iz": args.iz,
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
