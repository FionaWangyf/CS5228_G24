"""Train TabM to predict HDB monthly rent from the attached train.csv.

Usage:
    pip install tabm torch pandas numpy scikit-learn tqdm
    python train_tabm_hdb_rent.py --csv upload/train.csv
"""
from __future__ import annotations

import argparse
import copy
import json
from math import ceil
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn.functional as F
from tabm import TabM
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import load_data


CAT_COLS = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL"]
NUM_COLS = [
    "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE", "PROPERTY_AGE_AT_APPROVAL",
    "APPROVAL_YEAR", "APPROVAL_MONTH", "APPROVAL_MONTH_SIN", "APPROVAL_MONTH_COS",
]
TARGET = "MONTHLY_RENT"


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["RENT_APPROVAL_DATE"], errors="coerce")
    df["APPROVAL_YEAR"] = dt.dt.year.astype(float)
    df["APPROVAL_MONTH"] = dt.dt.month.astype(float)
    df["APPROVAL_MONTH_SIN"] = np.sin(2 * np.pi * df["APPROVAL_MONTH"] / 12)
    df["APPROVAL_MONTH_COS"] = np.cos(2 * np.pi * df["APPROVAL_MONTH"] / 12)
    df["PROPERTY_AGE_AT_APPROVAL"] = df["APPROVAL_YEAR"] - df["LEASE_COMMENCE_DATE"]
    for col in CAT_COLS:
        df[col] = df[col].fillna("__MISSING__").astype(str).str.strip().str.lower()
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", default=None,
        help="Optional training CSV; defaults to the repository's shared train.csv split",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--out", default="tabm_hdb_rent.pt")
    parser.add_argument("--no-progress", action="store_true", help="Disable training progress bars")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("--epochs, --batch-size, and --patience must all be positive")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_df, valid_df = load_data(args.csv or PROJECT_ROOT / "data" / "train.csv")
    parts = {"train": train_df, "val": valid_df}
    if any(part.empty for part in parts.values()):
        raise ValueError("The shared train/validation split contains an empty partition")
    if any(part[TARGET].isna().any() for part in parts.values()):
        raise ValueError(f"{TARGET} contains missing values in the shared split")
    print("Rows:", {k: len(v) for k, v in parts.items()})
    print("Date spans:", {
        k: (v.RENT_APPROVAL_DATE.min(), v.RENT_APPROVAL_DATE.max())
        for k, v in parts.items()
    })

    feature_parts = {name: make_features(part) for name, part in parts.items()}
    for name, features in feature_parts.items():
        missing = features[NUM_COLS].isna().sum()
        if missing.any():
            raise ValueError(
                f"Missing numeric TabM features in {name}: {missing[missing > 0].to_dict()}"
            )

    # Fit every transform on training data only. Unknown future categories get index 0.
    category_maps: dict[str, dict[str, int]] = {}
    for col in CAT_COLS:
        values = sorted(parts["train"][col].unique())
        category_maps[col] = {value: i + 1 for i, value in enumerate(values)}
    cat_cardinalities = [len(category_maps[c]) + 1 for c in CAT_COLS]

    x_scaler = QuantileTransformer(
        n_quantiles=min(1000, max(10, len(parts["train"]) // 30)),
        output_distribution="normal", random_state=seed,
    )
    x_scaler.fit(feature_parts["train"][NUM_COLS].to_numpy(dtype=np.float32))
    y_mean = float(parts["train"][TARGET].mean())
    y_std = float(parts["train"][TARGET].std()) or 1.0

    arrays = {}
    for name, part in parts.items():
        features = feature_parts[name]
        x_num = x_scaler.transform(features[NUM_COLS].to_numpy(dtype=np.float32)).astype(np.float32)
        x_cat = np.column_stack([
            features[col].map(category_maps[col]).fillna(0).to_numpy(dtype=np.int64)
            for col in CAT_COLS
        ])
        y = ((part[TARGET].to_numpy(dtype=np.float32) - y_mean) / y_std).astype(np.float32)
        arrays[name] = (torch.from_numpy(x_num), torch.from_numpy(x_cat), torch.from_numpy(y))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu (this PyTorch installation does not have CUDA available)")
    model = TabM.make(
        n_num_features=len(NUM_COLS), cat_cardinalities=cat_cardinalities, d_out=1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-4)

    def predict(part: str) -> np.ndarray:
        model.eval()
        x_num, x_cat, _ = arrays[part]
        outs = []
        with torch.inference_mode():
            for start in range(0, len(x_num), 8192):
                pred = model(x_num[start:start+8192].to(device), x_cat[start:start+8192].to(device))
                # TabM returns (batch, k, 1); average its k predictions at inference.
                outs.append(pred.squeeze(-1).mean(dim=1).cpu().numpy())
        return np.concatenate(outs) * y_std + y_mean

    best_rmse = float("inf")
    best_state = None
    stale = 0
    batches_per_epoch = ceil(len(arrays["train"][2]) / args.batch_size)
    epoch_progress = tqdm(
        range(1, args.epochs + 1),
        desc="Training TabM",
        unit="epoch",
        disable=args.no_progress,
    )
    for epoch in epoch_progress:
        model.train()
        x_num, x_cat, y = arrays["train"]
        order = torch.randperm(len(y))
        batch_losses = []
        batch_progress = tqdm(
            order.split(args.batch_size),
            total=batches_per_epoch,
            desc=f"Epoch {epoch:03d} batches",
            unit="batch",
            leave=False,
            disable=args.no_progress,
        )
        for idx in batch_progress:
            xn, xc, yt = x_num[idx].to(device), x_cat[idx].to(device), y[idx].to(device)
            pred = model(xn, xc).squeeze(-1)  # (batch, k)
            # Train each of the k predictions independently; do not average before loss.
            loss = F.mse_loss(pred, yt[:, None].expand_as(pred))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())
            batch_progress.set_postfix(loss=f"{loss.item():.4f}")

        y_val = parts["val"][TARGET].to_numpy(dtype=np.float32)
        pred_val = predict("val")
        rmse = float(sklearn.metrics.root_mean_squared_error(y_val, pred_val))
        epoch_progress.set_postfix(
            train_loss=f"{np.mean(batch_losses):.4f}",
            val_RMSE=f"${rmse:,.0f}",
        )
        if args.no_progress:
            print(f"epoch={epoch:03d} val_RMSE=${rmse:,.0f}")
        else:
            epoch_progress.write(f"epoch={epoch:03d} val_RMSE=${rmse:,.0f}")
        if rmse < best_rmse:
            best_rmse, stale = rmse, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    y_valid = parts["val"][TARGET].to_numpy(dtype=np.float32)
    pred_valid = predict("val")
    print("VALIDATION metrics (same fixed split as model_comparison):")
    print(f"MAE  = ${sklearn.metrics.mean_absolute_error(y_valid, pred_valid):,.0f}")
    print(f"RMSE = ${sklearn.metrics.root_mean_squared_error(y_valid, pred_valid):,.0f}")
    print(f"R2   = {sklearn.metrics.r2_score(y_valid, pred_valid):.4f}")

    out = Path(args.out)
    torch.save({
        "state_dict": model.state_dict(),
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "category_maps": category_maps,
        "cat_cardinalities": cat_cardinalities,
        "y_mean": y_mean,
        "y_std": y_std,
        "x_scaler": x_scaler,
        "best_val_rmse": best_rmse,
        "train_date_span": (parts["train"].RENT_APPROVAL_DATE.min(), parts["train"].RENT_APPROVAL_DATE.max()),
        "validation_date_span": (parts["val"].RENT_APPROVAL_DATE.min(), parts["val"].RENT_APPROVAL_DATE.max()),
    }, out)
    print(f"Saved model and preprocessing metadata to {out}")
    print("Example validation predictions:")
    for actual, pred in zip(y_valid[:5], pred_valid[:5]):
        print(f"actual=${actual:,.0f}  predicted=${pred:,.0f}")


if __name__ == "__main__":
    main()
