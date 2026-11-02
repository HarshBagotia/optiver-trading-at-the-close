from __future__ import annotations

import pandas as pd


def split_by_date(df: pd.DataFrame, last_dates: int, date_col: str = "date_id"):
    """Hold out the last `last_dates` trading days.

    Random row splits leak heavily here: neighbouring snapshots of the same stock-day share
    almost the same features and overlapping 60 s targets, so a random split mostly measures
    memorisation. Splitting by whole days mirrors how the model is used: trained on the
    past, scored on days it has never seen.
    """
    dates = sorted(df[date_col].unique())
    if last_dates < 1 or last_dates >= len(dates):
        raise ValueError(f"last_dates={last_dates} must be between 1 and {len(dates) - 1}")
    cutoff = dates[-last_dates]
    before, after = df[df[date_col] < cutoff], df[df[date_col] >= cutoff]
    return before.reset_index(drop=True), after.reset_index(drop=True)
