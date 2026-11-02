from dataclasses import replace

import pytest

from closing_auction.config import TrainConfig
from closing_auction.data import make_synthetic_data, validate_schema
from closing_auction.train import train


@pytest.fixture(scope="session")
def small_df():
    return validate_schema(make_synthetic_data(n_stocks=30, n_dates=16, seed=7))


@pytest.fixture(scope="session")
def trained_model_dir(small_df, tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    cfg = replace(
        TrainConfig(),
        artifacts_dir=out,
        holdout_dates=4,
        early_stopping_dates=3,
        num_boost_round=400,
        early_stopping_rounds=30,
        min_data_in_leaf=50,
    )
    return out, train(cfg, df=small_df)
