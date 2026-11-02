import numpy as np
import pandas as pd

from closing_auction.features import FEATURES, build_features


def _one_auction(df):
    return df[(df["stock_id"] == 3) & (df["date_id"] == 2)].reset_index(drop=True)


def test_no_look_ahead(small_df):
    """Features at time t must not change when later snapshots are removed."""
    day = _one_auction(small_df)
    full = build_features(day)
    for t in (0, 60, 300, 400):
        past = day[day["seconds_in_bucket"] <= t]
        partial = build_features(past)
        pd.testing.assert_frame_equal(partial.loc[past.index], full.loc[past.index])


def test_input_order_does_not_matter(small_df):
    sample = small_df[small_df["date_id"] == 1].reset_index(drop=True)
    shuffled = sample.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(build_features(shuffled).loc[sample.index], build_features(sample))


def test_lags_do_not_cross_stocks_or_days(small_df):
    feats = build_features(small_df)
    first = small_df["seconds_in_bucket"] == 0
    assert feats.loc[first, "wap_ret_1_bps"].isna().all()


def test_feature_columns_and_values(small_df):
    feats = build_features(small_df.head(200))
    assert list(feats.columns) == FEATURES
    li = feats["liquidity_imbalance"]
    assert li.between(-1, 1).all()
    assert np.isfinite(feats["spread_bps"]).all() and (feats["spread_bps"] >= 0).all()
