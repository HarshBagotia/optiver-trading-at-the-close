"""Train, evaluate and save the closing-auction forecaster.

Validation protocol (splits by trading day, never random):

    |------------ fit ------------|-- early stop --|----- holdout -----|
    |<-------------- training days --------------->|

1. Train on `fit`, early-stop on `early stop` to choose the number of trees.
2. Refit on all training days with that number of trees.
3. Score the untouched `holdout` days exactly once; report MAE next to naive baselines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import TrainConfig
from .data import TARGET, load_data, reduce_memory
from .features import CATEGORICAL, FEATURES, build_features
from .split import split_by_date
from .utils import file_sha256, set_seed

log = logging.getLogger(__name__)


def lgb_params(cfg: TrainConfig) -> dict:
    params = {
        "objective": "regression_l1",  # optimise MAE directly - the competition metric
        "metric": "l1",
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": 1,
        "lambda_l2": cfg.lambda_l2,
        "max_bin": cfg.max_bin,
        "seed": cfg.seed,
        "verbosity": -1,
        "device_type": cfg.device,
    }
    if cfg.device == "cpu":
        # Bit-for-bit reproducible on CPU; GPU histogram building is not deterministic.
        params.update(deterministic=True, force_row_wise=True)
    return params


def _dataset(df: pd.DataFrame, reference=None) -> lgb.Dataset:
    return lgb.Dataset(
        build_features(df),
        label=df[TARGET].to_numpy(),
        categorical_feature=CATEGORICAL,
        reference=reference,
        free_raw_data=True,
    )


def fit_booster(
    train_df: pd.DataFrame,
    cfg: TrainConfig,
    early_stop_df: pd.DataFrame | None = None,
    num_boost_round: int | None = None,
) -> lgb.Booster:
    dtrain = _dataset(train_df)
    valid_sets, callbacks = [], [lgb.log_evaluation(0)]
    if early_stop_df is not None:
        valid_sets.append(_dataset(early_stop_df, reference=dtrain))
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))
    return lgb.train(
        lgb_params(cfg),
        dtrain,
        num_boost_round=num_boost_round or cfg.num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def evaluate(booster: lgb.Booster, train_df: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, float]:
    """Holdout MAE of the model and of naive baselines it has to beat."""
    y = holdout[TARGET].to_numpy(dtype=float)
    preds = {
        "model": booster.predict(build_features(holdout)),
        "zero": np.zeros_like(y),  # "no move relative to the index"
        "train_median": np.full_like(y, float(np.median(train_df[TARGET]))),
    }
    return {f"mae_{name}": round(float(np.mean(np.abs(y - p))), 4) for name, p in preds.items()}


def feature_importance(booster: lgb.Booster, top: int = 15) -> list[dict]:
    gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(-gain)[:top]
    total = gain.sum() or 1.0
    names = booster.feature_name()
    return [{"feature": names[i], "gain_share": round(float(gain[i] / total), 4)} for i in order]


def train(cfg: TrainConfig, df: pd.DataFrame | None = None) -> dict:
    set_seed(cfg.seed)
    if df is None:
        df, memory = load_data(cfg.data_path)
        data_hash = file_sha256(cfg.data_path)
    else:
        df, memory = reduce_memory(df)
        data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy()).hexdigest()

    train_days, holdout = split_by_date(df, cfg.holdout_dates)
    fit_days, early_stop = split_by_date(train_days, cfg.early_stopping_dates)

    probe = fit_booster(fit_days, cfg, early_stop_df=early_stop)
    best_iteration = probe.best_iteration or cfg.num_boost_round
    log.info("early stopping picked %d trees", best_iteration)

    booster = fit_booster(train_days, cfg, num_boost_round=best_iteration)
    metrics = evaluate(booster, train_days, holdout)

    config_dict = cfg.to_dict()
    # Same data + hyper-parameters + feature set => same version, wherever the files are written.
    hparams = {k: v for k, v in config_dict.items() if k not in ("data_path", "artifacts_dir")}
    hparams["feature_set"] = {"features": FEATURES, "categorical": CATEGORICAL}
    version_src = json.dumps(hparams, sort_keys=True) + data_hash

    def span(x: pd.DataFrame) -> list[int]:
        return [int(x["date_id"].min()), int(x["date_id"].max())]

    metadata = {
        "model_version": hashlib.sha256(version_src.encode()).hexdigest()[:12],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config_dict,
        "data": {
            "rows": len(df),
            "stocks": int(df["stock_id"].nunique()),
            "dates": int(df["date_id"].nunique()),
            "sha256": data_hash,
            "memory": memory,
        },
        "splits": {"fit": span(fit_days), "early_stop": span(early_stop), "holdout": span(holdout)},
        "best_iteration": best_iteration,
        "features": FEATURES,
        "metrics": metrics,
        "top_features": feature_importance(booster),
        "library_versions": {"lightgbm": lgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    save_artifacts(cfg.artifacts_dir, booster, metadata)
    _maybe_log_mlflow(cfg.artifacts_dir, metadata)
    return metadata


def save_artifacts(out_dir: Path, booster: lgb.Booster, metadata: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "model.txt"))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _maybe_log_mlflow(out_dir: Path, metadata: dict) -> None:
    """Log the run to MLflow when MLFLOW_TRACKING_URI is set (optional dependency)."""
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
    except ImportError:
        log.warning("MLFLOW_TRACKING_URI is set but mlflow is not installed; skipping")
        return
    with mlflow.start_run(run_name=metadata["model_version"]):
        mlflow.log_params(metadata["config"])
        mlflow.log_param("data_sha256", metadata["data"]["sha256"])
        mlflow.log_metrics(metadata["metrics"])
        mlflow.log_artifacts(str(out_dir))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train the closing-auction forecaster")
    parser.add_argument("--data", type=Path, default=TrainConfig.data_path)
    parser.add_argument("--out", type=Path, default=TrainConfig.artifacts_dir)
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default=TrainConfig.device)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--holdout-dates", type=int, default=TrainConfig.holdout_dates)
    args = parser.parse_args()

    cfg = replace(
        TrainConfig(),
        data_path=args.data,
        artifacts_dir=args.out,
        device=args.device,
        seed=args.seed,
        holdout_dates=args.holdout_dates,
    )
    meta = train(cfg)
    print(json.dumps({"model_version": meta["model_version"], **meta["metrics"],
                      "memory": meta["data"]["memory"]}, indent=2))


if __name__ == "__main__":
    main()
