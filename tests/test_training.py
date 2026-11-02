from dataclasses import replace

from closing_auction.config import TrainConfig
from closing_auction.predict import AuctionForecaster
from closing_auction.train import train


def test_model_beats_naive_baselines(trained_model_dir):
    _, meta = trained_model_dir
    m = meta["metrics"]
    assert m["mae_model"] < m["mae_zero"]
    assert m["mae_model"] < m["mae_train_median"]


def test_artifacts_load_and_splits_are_ordered(trained_model_dir):
    out, meta = trained_model_dir
    assert AuctionForecaster.load(out).version == meta["model_version"]
    s = meta["splits"]
    assert s["fit"][1] < s["early_stop"][0] <= s["early_stop"][1] < s["holdout"][0]
    assert meta["data"]["memory"]["reduction_pct"] > 0


def test_training_is_reproducible(small_df, tmp_path):
    cfg = replace(
        TrainConfig(), holdout_dates=4, early_stopping_dates=3, num_boost_round=50,
        early_stopping_rounds=10, min_data_in_leaf=50,
    )
    a = train(replace(cfg, artifacts_dir=tmp_path / "a"), df=small_df)
    b = train(replace(cfg, artifacts_dir=tmp_path / "b"), df=small_df)
    assert a["metrics"] == b["metrics"] and a["model_version"] == b["model_version"]
    assert (tmp_path / "a" / "model.txt").read_text() == (tmp_path / "b" / "model.txt").read_text()
