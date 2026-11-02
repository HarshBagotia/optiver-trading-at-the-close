import numpy as np
import pandas as pd
import pytest

from closing_auction.data import SchemaError, make_synthetic_data, reduce_memory, validate_schema
from closing_auction.split import split_by_date


def test_synthetic_data_matches_competition_layout():
    df = make_synthetic_data(n_stocks=5, n_dates=2, seed=1)
    assert len(df) == 5 * 2 * 55
    assert df["seconds_in_bucket"].max() == 540
    early = df["seconds_in_bucket"] < 300
    assert df.loc[early, "far_price"].isna().all() and df.loc[~early, "far_price"].notna().all()
    # target is relative to the index: it averages to ~0 across stocks at each timestamp
    assert df.groupby(["date_id", "seconds_in_bucket"])["target"].mean().abs().max() < 1e-3


def test_reduce_memory_shrinks_and_preserves_values(small_df):
    reduced, report = reduce_memory(small_df)
    assert report["after_mb"] < report["before_mb"]
    assert report["reduction_pct"] >= 30
    assert reduced["wap"].dtype == np.float32
    assert reduced["seconds_in_bucket"].dtype.itemsize <= 2
    np.testing.assert_allclose(reduced["wap"], small_df["wap"], rtol=1e-6)
    np.testing.assert_allclose(reduced["matched_size"], small_df["matched_size"], rtol=1e-6)


def test_split_by_date_is_disjoint_and_ordered(small_df):
    train, holdout = split_by_date(small_df, last_dates=4)
    assert train["date_id"].max() < holdout["date_id"].min()
    assert holdout["date_id"].nunique() == 4
    assert len(train) + len(holdout) == len(small_df)


def test_schema_rejects_duplicates_and_missing_columns(small_df):
    with pytest.raises(SchemaError, match="duplicate"):
        validate_schema(pd.concat([small_df.head(3), small_df.head(1)]))
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_schema(small_df.drop(columns=["wap"]))


def test_schema_drops_rows_without_target(small_df):
    df = small_df.head(10).copy()
    df.loc[0, "target"] = np.nan
    assert len(validate_schema(df)) == 9
