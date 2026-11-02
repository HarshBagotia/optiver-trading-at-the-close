"""Training configuration. Every run is fully described by this object + the data hash."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path = Path("data/train.csv")
    artifacts_dir: Path = Path("artifacts")
    seed: int = 42

    # Splits are by trading day (date_id): the last `holdout_dates` days are never used for
    # training; the `early_stopping_dates` days before them pick the number of trees.
    holdout_dates: int = 8
    early_stopping_dates: int = 6

    # "gpu" (OpenCL) / "cuda" need a GPU-enabled LightGBM build; "cpu" works everywhere.
    device: Literal["cpu", "gpu", "cuda"] = "cpu"

    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    max_bin: int = 255

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_path"] = str(self.data_path)
        d["artifacts_dir"] = str(self.artifacts_dir)
        return d
