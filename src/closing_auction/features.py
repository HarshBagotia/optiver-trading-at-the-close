"""Order-book feature engineering.

Everything here is a pure function of the snapshots passed in - no fitted state - so the
exact same code runs in training and in the API. Features for a row only ever use that row
and *earlier* snapshots of the same stock on the same day (no look-ahead), which
`tests/test_features.py` checks explicitly.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

PRICES = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
RAW = [
    "stock_id",
    "seconds_in_bucket",
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
BOOK = [
    "volume",
    "mid_price",
    "spread_bps",
    "liquidity_imbalance",
    "log_size_ratio",
    "matched_imbalance",
    "signed_imbalance",
    "imbalance_ratio",
    "wap_vs_mid_bps",
]
PAIRS = [f"{a}_{b}_imb" for a, b in combinations(PRICES, 2)]
RETURN_COLS = ["wap", "reference_price"]  # relative change, in bps
DIFF_COLS = ["liquidity_imbalance", "imbalance_ratio", "matched_size"]
LAGS = (1, 2, 3, 6)
DYNAMICS = [f"{c}_ret_{k}_bps" for c in RETURN_COLS for k in LAGS] + [
    f"{c}_diff_{k}" for c in DIFF_COLS for k in LAGS
]
FEATURES = RAW + BOOK + PAIRS + DYNAMICS
# stock_id is used as a plain numeric feature: as a 200-level categorical LightGBM spent ~40%
# of its gain on it and generalised worse (holdout MAE 12.45 vs 12.41 on the synthetic data).
CATEGORICAL: list[str] = []
_EPS = 1e-9


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature frame aligned to `df`'s index (input order does not matter)."""
    order = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"], kind="mergesort").index
    d = df.loc[order]
    f = pd.DataFrame(index=d.index)
    for c in RAW:
        f[c] = d[c]

    bid, ask = d["bid_price"].astype(float), d["ask_price"].astype(float)
    bid_size, ask_size = d["bid_size"].astype(float), d["ask_size"].astype(float)
    imb, matched = d["imbalance_size"].astype(float), d["matched_size"].astype(float)

    f["volume"] = bid_size + ask_size
    f["mid_price"] = (bid + ask) / 2
    f["spread_bps"] = (ask - bid) / f["mid_price"] * 1e4
    f["liquidity_imbalance"] = (bid_size - ask_size) / (f["volume"] + _EPS)
    f["log_size_ratio"] = np.log1p(bid_size) - np.log1p(ask_size)
    f["matched_imbalance"] = (imb - matched) / (imb + matched + _EPS)
    f["signed_imbalance"] = imb * d["imbalance_buy_sell_flag"]
    f["imbalance_ratio"] = f["signed_imbalance"] / (matched + _EPS)
    f["wap_vs_mid_bps"] = (d["wap"] / f["mid_price"] - 1) * 1e4

    for a, b in combinations(PRICES, 2):
        x, y = d[a].astype(float), d[b].astype(float)
        f[f"{a}_{b}_imb"] = (x - y) / (x + y)

    # Within-auction dynamics: rows are sorted by time inside each (stock, day) group, so
    # shift/diff only ever look backwards.
    groups = [d["stock_id"], d["date_id"]]
    for c in RETURN_COLS:
        g = d[c].astype(float).groupby(groups, sort=False)
        for k in LAGS:
            f[f"{c}_ret_{k}_bps"] = (d[c] / g.shift(k) - 1) * 1e4
    for c in DIFF_COLS:
        g = f[c].groupby(groups, sort=False)
        for k in LAGS:
            f[f"{c}_diff_{k}"] = g.diff(k)

    return f.loc[df.index, FEATURES]
