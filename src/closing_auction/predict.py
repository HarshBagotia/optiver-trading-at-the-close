from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import build_features

ARTIFACT_FILES = ("model.txt", "metadata.json")


class AuctionForecaster:
    """Loads a trained model and predicts 60 s index-relative moves (bps) for snapshots."""

    def __init__(self, booster: lgb.Booster, metadata: dict):
        self.booster = booster
        self.metadata = metadata

    @classmethod
    def load(cls, model_dir: Path) -> AuctionForecaster:
        model_dir = Path(model_dir)
        missing = [f for f in ARTIFACT_FILES if not (model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"missing artifacts in {model_dir}: {missing} - run `make train`")
        booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        metadata = json.loads((model_dir / "metadata.json").read_text())
        return cls(booster, metadata)

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    def predict(self, snapshots: pd.DataFrame) -> np.ndarray:
        """Snapshots may span several stocks and seconds; earlier seconds of the same
        stock-day are used for the lag features of later ones."""
        return self.booster.predict(build_features(snapshots))
