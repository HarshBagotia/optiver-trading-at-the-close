"""Loading, schema validation, memory optimisation and a synthetic data generator.

Real data: the Kaggle competition "Optiver - Trading at the Close"
(`kaggle competitions download -c optiver-trading-at-the-close`). It is not redistributed
here. The synthetic generator reproduces its schema (200 stocks x 55 ten-second snapshots of
the closing auction per day, target = 60 s future WAP move relative to an index, in bps) so
the pipeline, tests and CI run without it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

KEY_COLUMNS = ["stock_id", "date_id", "seconds_in_bucket"]
BOOK_COLUMNS = [
    "imbalance_size",
    "imbalance_buy_sell_flag",
    "reference_price",
    "matched_size",
    "far_price",
    "near_price",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
    "wap",
]
TARGET = "target"
NULLABLE = {"far_price", "near_price"}  # only published from 300 s into the auction
REQUIRED_COLUMNS = KEY_COLUMNS + BOOK_COLUMNS + [TARGET]


class SchemaError(ValueError):
    pass


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    non_null = [c for c in REQUIRED_COLUMNS if c not in NULLABLE and c != TARGET]
    bad = [c for c in non_null if df[c].isna().any()]
    if bad:
        raise SchemaError(f"null values in columns that must be populated: {bad}")
    if df.duplicated(KEY_COLUMNS).any():
        raise SchemaError("duplicate (stock_id, date_id, seconds_in_bucket) rows")
    # The real data has a handful of rows with a missing target; they cannot be learned from.
    return df[df[TARGET].notna()].reset_index(drop=True)


def reduce_memory(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Downcast numeric columns to the smallest dtype that holds them.

    Integers go to the smallest signed type that fits their range; floats go to float32,
    which keeps ~7 significant digits - plenty for prices around 1.0 and sizes, and halves
    their footprint. Returns the new frame and a before/after report.
    """
    before = df.memory_usage(deep=True).sum()
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            out[col] = s.astype(np.float32)
    after = out.memory_usage(deep=True).sum()
    report = {
        "before_mb": round(before / 2**20, 2),
        "after_mb": round(after / 2**20, 2),
        "reduction_pct": round(100 * (1 - after / before), 1),
    }
    return out, report


def load_data(path: Path) -> tuple[pd.DataFrame, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make data` or download the Kaggle data")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df = validate_schema(df)
    df, report = reduce_memory(df)
    log.info("memory %.1f MB -> %.1f MB (-%.1f%%)", report["before_mb"], report["after_mb"],
             report["reduction_pct"])
    return df, report


def make_synthetic_data(
    n_stocks: int = 200, n_dates: int = 40, seed: int = 42, steps: int = 55, horizon: int = 6
) -> pd.DataFrame:
    """Simulate closing-auction order books with a persistent order-flow signal.

    A hidden AR(1) order-flow process drives both the observable book (bid/ask size
    imbalance, auction imbalance side and size) and the drift of the price, so the future
    WAP move is partly predictable from the current book - as in the real market, most of
    it is noise.
    """
    rng = np.random.default_rng(seed)
    D, S, T, H = n_dates, n_stocks, steps, horizon
    shape = (D, S, T + H)

    rho = 0.9
    eps = rng.normal(0.0, 1.0, shape)
    flow = np.empty(shape)
    flow[..., 0] = eps[..., 0]
    for t in range(1, T + H):
        flow[..., t] = rho * flow[..., t - 1] + np.sqrt(1 - rho**2) * eps[..., t]

    vol = rng.uniform(2e-4, 8e-4, S)[None, :, None]  # per-stock volatility per 10 s step
    log_ret = 0.35 * vol * flow + vol * rng.normal(0.0, 1.0, shape)
    log_mid = np.cumsum(log_ret, axis=2)
    mid = (1.0 + rng.normal(0.0, 0.002, (D, S, 1))) * np.exp(log_mid - log_mid[..., :1])

    half_spread = (rng.uniform(3, 15, S)[None, :, None] * (1 + 0.2 * rng.random(shape))) / 2e4
    bid_price, ask_price = mid * (1 - half_spread), mid * (1 + half_spread)

    liq = np.tanh(0.6 * flow + rng.normal(0.0, 0.8, shape))
    depth = rng.lognormal(10.5, 0.8, S)[None, :, None] * rng.lognormal(0.0, 0.3, shape)
    bid_size, ask_size = depth * (1 + liq) / 2, depth * (1 - liq) / 2
    wap = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)

    # 60 s future WAP move of each stock, minus the equal-weighted index move, in bps
    fut = (wap[..., H:] / wap[..., :T] - 1.0) * 1e4
    target = fut - fut.mean(axis=1, keepdims=True)

    sl = (slice(None), slice(None), slice(0, T))
    signed = 0.5 * flow[sl] + rng.normal(0.0, 1.0, (D, S, T))
    flag = np.where(np.abs(signed) < 0.1, 0, np.sign(signed)).astype(int)
    progress = np.linspace(0.3, 1.0, T)[None, None, :]
    matched = rng.lognormal(15.0, 1.0, S)[None, :, None] * progress * rng.lognormal(0, 0.05, (D, S, T))
    imbalance = np.abs(signed) * matched * 0.05 * (flag != 0)
    reference = mid[sl] * (1 + flag * half_spread[sl] * 0.5)

    seconds = np.arange(T) * 10
    late = (seconds >= 300)[None, None, :]
    near = np.where(late, reference * (1 + rng.normal(0, 2e-4, (D, S, T))), np.nan)
    far = np.where(late, reference * (1 + rng.normal(0, 1e-3, (D, S, T))), np.nan)

    d_idx, s_idx, t_idx = np.meshgrid(np.arange(D), np.arange(S), np.arange(T), indexing="ij")
    cols = {
        "stock_id": s_idx,
        "date_id": d_idx,
        "seconds_in_bucket": t_idx * 10,
        "imbalance_size": imbalance,
        "imbalance_buy_sell_flag": flag,
        "reference_price": reference,
        "matched_size": matched,
        "far_price": far,
        "near_price": near,
        "bid_price": bid_price[sl],
        "bid_size": bid_size[sl],
        "ask_price": ask_price[sl],
        "ask_size": ask_size[sl],
        "wap": wap[sl],
        "target": target,
    }
    df = pd.DataFrame({k: v.reshape(-1) for k, v in cols.items()})
    df = df.sort_values(["date_id", "seconds_in_bucket", "stock_id"], kind="mergesort")
    df["time_id"] = df["date_id"] * T + df["seconds_in_bucket"] // 10
    df["row_id"] = (
        df["date_id"].astype(str)
        + "_"
        + df["seconds_in_bucket"].astype(str)
        + "_"
        + df["stock_id"].astype(str)
    )
    price_cols = ["reference_price", "far_price", "near_price", "bid_price", "ask_price", "wap"]
    df[price_cols] = df[price_cols].round(6)
    df[["imbalance_size", "matched_size", "bid_size", "ask_size"]] = df[
        ["imbalance_size", "matched_size", "bid_size", "ask_size"]
    ].round(2)
    df["target"] = df["target"].round(4)
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic closing-auction data")
    parser.add_argument("--out", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--stocks", type=int, default=200)
    parser.add_argument("--dates", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = make_synthetic_data(args.stocks, args.dates, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df):,} rows to {args.out} (target std {df['target'].std():.2f} bps)")


if __name__ == "__main__":
    main()
