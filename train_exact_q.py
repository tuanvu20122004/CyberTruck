#!/usr/bin/env python3
"""
Train Model 2: pure Exact-Q imitation using dQ/du from the Lagrange multiplier.

Core gradient:
    grad_theta L_Q = E[ dQ/du(x, u_theta) * grad_theta pi_theta(x) ]

The C++ pybind module `exactq_mpc` returns dQ/du_deg from the dual variable of
constraint u0 - u_fixed = 0. This script uses the surrogate scalar loss:
    loss_q = mean( stopgrad(dQ/du_deg) * u_theta_deg )
so PyTorch receives exactly dQ/du as d(loss)/d(action).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    import exactq_mpc
except ImportError as exc:
    exactq_mpc = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


FEATURE_COLUMNS = [
    "lateral_deviation",
    "yaw_angle",
    "curvature_0",
    "curvature_1",
    "curvature_2",
    "curvature_3",
    "velocity",
    "prev_steering",
]


class PolicyMLP(nn.Module):
    def __init__(self, model_json: Dict):
        super().__init__()
        self.feature_columns = list(model_json["feature_columns"])
        if self.feature_columns != FEATURE_COLUMNS:
            raise ValueError(
                f"Feature order mismatch. JSON has {self.feature_columns}, "
                f"expected {FEATURE_COLUMNS}"
            )

        normalization = model_json["normalization"]
        self.register_buffer(
            "feature_mean",
            torch.tensor(normalization["feature_mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "feature_std",
            torch.tensor(normalization["feature_std"], dtype=torch.float32),
        )
        self.register_buffer(
            "target_mean",
            torch.tensor(float(normalization["target_mean"]), dtype=torch.float32),
        )
        self.register_buffer(
            "target_std",
            torch.tensor(float(normalization["target_std"]), dtype=torch.float32),
        )

        self.activation_name = model_json.get("activation", "tanh")
        if self.activation_name not in {"tanh", "relu"}:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

        self.layers = nn.ModuleList()
        for layer in model_json["layers"]:
            w = torch.tensor(layer["weight"], dtype=torch.float32)
            b = torch.tensor(layer["bias"], dtype=torch.float32)
            linear = nn.Linear(w.shape[1], w.shape[0])
            with torch.no_grad():
                linear.weight.copy_(w)
                linear.bias.copy_(b)
            self.layers.append(linear)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return torch.relu(x)
        return torch.tanh(x)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        std = torch.where(torch.abs(self.feature_std) < 1e-8, torch.ones_like(self.feature_std), self.feature_std)
        x = (x_raw - self.feature_mean) / std
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 != len(self.layers):
                x = self.activation(x)
        # The network output is normalized target; runtime PolicyModel denormalizes it.
        return x.squeeze(-1) * self.target_std + self.target_mean

    def export_json(self, template_json: Dict) -> Dict:
        out = dict(template_json)
        out["feature_columns"] = FEATURE_COLUMNS
        out["activation"] = self.activation_name
        out["normalization"] = {
            "feature_mean": self.feature_mean.detach().cpu().numpy().astype(float).tolist(),
            "feature_std": self.feature_std.detach().cpu().numpy().astype(float).tolist(),
            "target_mean": float(self.target_mean.detach().cpu().item()),
            "target_std": float(self.target_std.detach().cpu().item()),
        }
        layers = []
        for layer in self.layers:
            layers.append(
                {
                    "weight": layer.weight.detach().cpu().numpy().astype(float).tolist(),
                    "bias": layer.bias.detach().cpu().numpy().astype(float).tolist(),
                }
            )
        out["layers"] = layers
        return out


def load_model_json(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    # Standard JSON is expected. This small guard helps if the file was saved with a UTF-8 BOM.
    text = text.lstrip("\ufeff")
    return json.loads(text)


def save_model_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def load_dataset(csv_path: Path, horizon: int, curvature_padding: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required policy columns: {missing}")

    if "is_valid" in df.columns:
        df = df[df["is_valid"].astype(int) == 1].copy()

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)

    policy_x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    # MPC state matrix consumed by exactq_mpc:
    # [ey, yaw, curvature_0..curvature_{N-1}, velocity]
    mpc_cols: List[np.ndarray] = [
        df["lateral_deviation"].to_numpy(dtype=np.float64),
        df["yaw_angle"].to_numpy(dtype=np.float64),
    ]

    available_curvatures = [c for c in [f"curvature_{i}" for i in range(horizon)] if c in df.columns]
    if not available_curvatures:
        raise ValueError("Dataset must contain at least curvature_0")

    last_available = available_curvatures[-1]
    for i in range(horizon):
        col = f"curvature_{i}"
        if col in df.columns:
            mpc_cols.append(df[col].to_numpy(dtype=np.float64))
        elif curvature_padding == "repeat_last":
            mpc_cols.append(df[last_available].to_numpy(dtype=np.float64))
        elif curvature_padding == "zero":
            mpc_cols.append(np.zeros(len(df), dtype=np.float64))
        else:
            raise ValueError(f"Unknown curvature_padding: {curvature_padding}")

    mpc_cols.append(df["velocity"].to_numpy(dtype=np.float64))
    mpc_state = np.stack(mpc_cols, axis=1).astype(np.float64)

    finite_policy = np.isfinite(policy_x).all(axis=1)
    finite_mpc = np.isfinite(mpc_state).all(axis=1)
    keep = finite_policy & finite_mpc

    return policy_x[keep], mpc_state[keep]


def apply_action_limit(u_deg: torch.Tensor, limit_deg: float, mode: str) -> torch.Tensor:
    if mode == "none":
        return u_deg
    if mode == "clamp":
        return torch.clamp(u_deg, -limit_deg, limit_deg)
    if mode == "tanh":
        return limit_deg * torch.tanh(u_deg / limit_deg)
    raise ValueError(f"Unknown action limit mode: {mode}")


def estimate_dual_sign(
    evaluator,
    mpc_state: np.ndarray,
    u_deg: np.ndarray,
    eps_deg: float,
    max_samples: int,
) -> float:
    n = min(len(u_deg), max_samples)
    if n == 0:
        return 1.0

    idx = np.random.choice(len(u_deg), size=n, replace=False)
    s = mpc_state[idx]
    u = u_deg[idx].astype(np.float64)

    res = evaluator.eval_batch(s, u)
    res_p = evaluator.eval_batch(s, u + eps_deg)
    res_m = evaluator.eval_batch(s, u - eps_deg)

    q = np.asarray(res["q"], dtype=np.float64)
    qp = np.asarray(res_p["q"], dtype=np.float64)
    qm = np.asarray(res_m["q"], dtype=np.float64)
    dq = np.asarray(res["dq_du_deg"], dtype=np.float64)
    valid = (
        np.asarray(res["valid"], dtype=np.int32).astype(bool)
        & np.asarray(res_p["valid"], dtype=np.int32).astype(bool)
        & np.asarray(res_m["valid"], dtype=np.int32).astype(bool)
        & np.isfinite(q)
        & np.isfinite(qp)
        & np.isfinite(qm)
        & np.isfinite(dq)
    )

    if valid.sum() < max(3, n // 5):
        print("[WARN] Not enough valid samples for finite-difference sign check; keeping sign +1.")
        return 1.0

    fd = (qp[valid] - qm[valid]) / (2.0 * eps_deg)
    dqv = dq[valid]
    dot = float(np.median(fd * dqv))
    corr_num = float(np.dot(fd, dqv))
    corr_den = float(np.linalg.norm(fd) * np.linalg.norm(dqv) + 1e-12)
    corr = corr_num / corr_den

    print(f"[CHECK] finite-difference sign: median(fd*dq)={dot:.6e}, cosine={corr:.4f}")
    if dot < 0.0:
        print("[CHECK] Dual sign appears flipped. Applying sign factor -1 in Python training.")
        return -1.0
    print("[CHECK] Dual sign is consistent. Applying sign factor +1.")
    return 1.0


def train(args: argparse.Namespace) -> None:
    if exactq_mpc is None:
        raise RuntimeError(
            "Cannot import exactq_mpc. Build exactq_bindings.cpp first and ensure the module "
            "is on PYTHONPATH. Original import error: " + repr(_IMPORT_ERROR)
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    model_json = load_model_json(Path(args.init_json))
    model = PolicyMLP(model_json).to(device)

    policy_x, mpc_state = load_dataset(Path(args.dataset), args.horizon, args.curvature_padding)
    print(f"[DATA] samples={len(policy_x)}, policy_dim={policy_x.shape[1]}, mpc_state_dim={mpc_state.shape[1]}")
    if len(policy_x) == 0:
        raise RuntimeError("No valid samples after filtering")

    evaluator = exactq_mpc.ExactQEvaluator(
        horizon=args.horizon,
        q1=args.q1,
        q2=args.q2,
        r=args.r,
        wheelbase=args.wheelbase,
        mass=args.mass,
        lf=args.lf,
        lr=args.lr_rear,
        caf=args.caf,
        car=args.car,
        iz=args.iz,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Initial action for dual sign check.
    with torch.no_grad():
        check_idx = np.random.choice(len(policy_x), size=min(len(policy_x), args.sign_check_samples), replace=False)
        x_check = torch.tensor(policy_x[check_idx], dtype=torch.float32, device=device)
        u_check = model(x_check)
        u_check = apply_action_limit(u_check, args.action_limit_deg, args.action_limit_mode)
        u_check_np = u_check.detach().cpu().numpy().astype(np.float64)

    sign_factor = 1.0
    if args.dq_sign == "auto":
        sign_factor = estimate_dual_sign(
            evaluator,
            mpc_state[check_idx],
            u_check_np,
            args.fd_eps_deg,
            args.sign_check_samples,
        )
    elif args.dq_sign == "minus":
        sign_factor = -1.0

    n = len(policy_x)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        order = np.random.permutation(n)
        epoch_loss = []
        epoch_q = []
        epoch_dq_abs = []
        epoch_valid = 0

        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            x_np = policy_x[idx]
            s_np = mpc_state[idx]

            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            u_pred = model(x)
            u_limited = apply_action_limit(u_pred, args.action_limit_deg, args.action_limit_mode)

            # The QP solver is external/non-differentiable. We only need dQ/du at the current action.
            u_for_solver = u_limited.detach().cpu().numpy().astype(np.float64)
            qres = evaluator.eval_batch(s_np, u_for_solver)

            q_np = np.asarray(qres["q"], dtype=np.float64)
            dq_np = sign_factor * np.asarray(qres["dq_du_deg"], dtype=np.float64)
            valid_np = np.asarray(qres["valid"], dtype=np.int32).astype(bool)
            valid_np &= np.isfinite(q_np) & np.isfinite(dq_np)

            if valid_np.sum() == 0:
                continue

            valid_mask = torch.tensor(valid_np, dtype=torch.bool, device=device)
            dq = torch.tensor(dq_np, dtype=torch.float32, device=device).detach()
            dq = torch.clamp(dq, -args.max_abs_dq, args.max_abs_dq)

            # Surrogate scalar. Its derivative wrt u_limited is exactly dq.
            loss_q = torch.mean(dq[valid_mask] * u_limited[valid_mask])

            optimizer.zero_grad(set_to_none=True)
            loss_q.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss.append(float(loss_q.detach().cpu().item()))
            epoch_q.append(float(np.mean(q_np[valid_np])))
            epoch_dq_abs.append(float(np.mean(np.abs(dq_np[valid_np]))))
            epoch_valid += int(valid_np.sum())
            global_step += 1

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            mean_loss = float(np.mean(epoch_loss)) if epoch_loss else math.nan
            mean_q = float(np.mean(epoch_q)) if epoch_q else math.nan
            mean_abs_dq = float(np.mean(epoch_dq_abs)) if epoch_dq_abs else math.nan
            print(
                f"[EPOCH {epoch:04d}] loss_surrogate={mean_loss:.6e} "
                f"mean_Q={mean_q:.6e} mean_abs_dQdu={mean_abs_dq:.6e} "
                f"valid_q={epoch_valid}"
            )

        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            ckpt_path = Path(args.output_json).with_suffix(f".epoch{epoch}.json")
            save_model_json(ckpt_path, model.export_json(model_json))
            print(f"[SAVE] checkpoint: {ckpt_path}")

    save_model_json(Path(args.output_json), model.export_json(model_json))
    print(f"[SAVE] exact-Q policy: {args.output_json}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pure Exact-Q policy using Lagrange multiplier dQ/du.")
    p.add_argument("--dataset", required=True, help="CSV dataset with policy features")
    p.add_argument("--init-json", required=True, help="BC model JSON used as initialization")
    p.add_argument("--output-json", default="exact_q_train.json", help="Output Exact-Q model JSON")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--curvature-padding", choices=["repeat_last", "zero"], default="repeat_last")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-abs-dq", type=float, default=1e4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=0)

    p.add_argument("--action-limit-deg", type=float, default=28.0)
    p.add_argument("--action-limit-mode", choices=["clamp", "tanh", "none"], default="clamp")

    p.add_argument("--dq-sign", choices=["auto", "plus", "minus"], default="auto")
    p.add_argument("--fd-eps-deg", type=float, default=0.1)
    p.add_argument("--sign-check-samples", type=int, default=32)

    # Must match runtime MPC settings if you want fair evaluation.
    p.add_argument("--q1", type=float, default=1000.0)
    p.add_argument("--q2", type=float, default=50.0)
    p.add_argument("--r", type=float, default=5.0)
    p.add_argument("--wheelbase", type=float, default=0.2515)
    p.add_argument("--mass", type=float, default=2.3)
    p.add_argument("--lf", type=float, default=0.132)
    p.add_argument("--lr-rear", dest="lr_rear", type=float, default=0.12)
    p.add_argument("--caf", type=float, default=0.04)
    p.add_argument("--car", type=float, default=0.02)
    p.add_argument("--iz", type=float, default=0.04)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
