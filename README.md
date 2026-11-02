# Optiver: Trading at the Close, Closing-Auction Price Forecasting

[![CI](https://github.com/HarshBagotia/optiver-trading-at-the-close/actions/workflows/ci.yml/badge.svg)](https://github.com/HarshBagotia/optiver-trading-at-the-close/actions/workflows/ci.yml)

Forecasts how each of 200 Nasdaq stocks will move over the next 60 seconds of the **closing
auction**, relative to a synthetic index, using order-book and auction-imbalance snapshots
taken every 10 seconds. The pipeline runs end to end: memory-optimized loading,
no-look-ahead order-book features, a LightGBM model optimizing **MAE** (GPU-capable),
day-based validation against naive baselines, and a **FastAPI** service packaged with
**Docker** and tested in **CI**.

| | |
|---|---|
| **Competition result** | **5.455 MAE** |
| Memory optimization | **~30% lower memory** on the competition data (dtype downcasting) |
| Model | LightGBM, `regression_l1` objective, `device_type=gpu`/`cuda` supported |
| Data | 200 stocks × 55 snapshots per auction × 481 trading days (~5.2M rows) |
| Stack | Python, Pandas, NumPy, LightGBM, FastAPI, Docker, GitHub Actions |

> **About the data.** The data belongs to the Kaggle competition
> [Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close)
> and is not redistributed here. The repo includes a **synthetic generator with the same
> schema** (same columns, 55 ten-second snapshots per auction, far/near prices only from
> 300 s, index-relative target in bps), so you can run, test and deploy everything without
> it. Scores on synthetic data cannot be compared with the competition score.

---

## Approach

**1. Target.** `target` is the 60-second future move of a stock's weighted average price (WAP)
minus the move of a synthetic index, in basis points. Most of it is noise, so the aim is to
pull a small, real signal out of the order book. LightGBM is trained directly on MAE
(`regression_l1`), the competition metric.

**2. Features** (`src/closing_auction/features.py`, 57 features)

| Group | Examples | Intuition |
|---|---|---|
| Raw book | prices, sizes, imbalance side, seconds into the auction | current state of the book |
| Book shape | spread (bps), **liquidity imbalance** `(bid_size − ask_size)/(bid_size + ask_size)`, log size ratio, WAP vs mid | buying vs selling pressure at the touch |
| Auction imbalance | signed imbalance, imbalance / matched size, matched imbalance | unfilled auction demand that must clear by the close |
| Price-pair imbalances | `(a − b)/(a + b)` for all 15 pairs of reference, far, near, bid, ask, WAP | where the auction is likely to clear relative to the market |
| Within-auction dynamics | WAP and reference-price returns over 1/2/3/6 snapshots; changes in liquidity imbalance, imbalance ratio and matched size | momentum and how pressure is building |

Top features by gain on the synthetic run: `reference_price_ret_6_bps`,
`liquidity_imbalance_diff_1`, `imbalance_ratio`, `liquidity_imbalance_diff_2`,
`ask_price_wap_imb`.

**3. Memory optimization** (`reduce_memory` in `src/closing_auction/data.py`). Integer
columns are downcast to the smallest type that fits and float64 becomes float32 (~7
significant digits, plenty for prices near 1.0). That cut memory by about 30% on the
competition data and by 40.9% (78.0 MB → 46.2 MB) on the synthetic data. This is what lets
the full dataset and all features fit in memory for GPU training.

**4. Model.** LightGBM with early stopping, plus `device_type="gpu"` or `"cuda"` for GPU
training (`--device gpu`; needs a GPU-enabled LightGBM build).

## Validation and leakage prevention

- **Splits by trading day, never by row.** Neighboring snapshots of the same stock-day have
  nearly identical features and overlapping 60-second targets, so a random row split mostly
  measures memorization. Whole days are held out:

  ```
  |------------ fit ------------|-- early stop --|----- holdout -----|
  ```

  The number of trees is chosen on the early-stopping days. The model is then refit on all
  training days and scored **once** on the untouched holdout days.
- **No look-ahead in features.** Lags and returns only look backward within the same stock
  and day. `tests/test_features.py` checks that a row's features stay identical when every
  later snapshot is deleted, that lags never cross stocks or days, and that input row order
  does not matter.
- **Same code in training and serving.** `build_features` is a pure function with no fitted
  state. An API test checks that predictions from the service match the offline model.

## Results on the bundled synthetic data

`make data && make train` (440k rows: 200 stocks × 40 days; holdout = last 8 days):

| Predictor | Holdout MAE (bps) |
|---|---|
| **LightGBM** | **12.41** |
| Always predict 0 (no move vs index) | 13.01 |
| Training-set median | 13.01 |

The gain over the naive baselines is about 5% (on the real data, top competition solutions
improved on the zero baseline by roughly 15%). That's normal for short-horizon returns: most
of the move is unpredictable, and the value is in the small, consistent edge.

## Reproducibility

- Seeds are fixed and LightGBM runs with `deterministic=True` on CPU. Dependencies are
  pinned. Two runs produce byte-identical models, and a test asserts this. (GPU histogram
  building is not bit-for-bit deterministic, so CPU is the reproducible reference.)
- `artifacts/metadata.json` records the config, the **SHA-256 of the data**, the memory
  report, the split day ranges, the tree count, the metrics, the top features and library
  versions. `model_version` is a hash of the hyper-parameters, feature set and data.
- Optional **MLflow** tracking: set `MLFLOW_TRACKING_URI` (and `pip install mlflow`).

## Quickstart

```bash
make install   # dependencies + package (editable)
make data      # synthetic data -> data/train.csv
make train     # train, evaluate, save to artifacts/
make test      # 21 unit + integration tests
make serve     # API on http://localhost:8000 (interactive docs at /docs)
```

With the real data:

```bash
kaggle competitions download -c optiver-trading-at-the-close && unzip optiver-trading-at-the-close.zip -d data/
python -m closing_auction.train --data data/train.csv --out artifacts --device gpu
```

With Docker:

```bash
make train && make docker-build && make docker-run
```

## Serving API

| Endpoint | Purpose | Caching |
|---|---|---|
| `POST /predict` | forecasts for a batch of snapshots | `no-store`: market data changes every 10 s, so a cached forecast is a wrong forecast |
| `GET /model/info` | version, splits, metrics, top features | `public, max-age=300` + `ETag` (returns 304 when unchanged) |
| `GET /health` | liveness and model status | `no-store` |

The service is **stateless**. Lag features come from the snapshots in the request, so send
each stock's earlier snapshots from the same auction to get full features. Missing history
becomes NaN, which the model handles exactly as it does at the start of every auction.

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @examples/predict_request.json
```

```json
{
  "model_version": "120506bd257c",
  "predictions": [
    {"stock_id": 7, "date_id": 39, "seconds_in_bucket": 260, "target_bps": -3.1683},
    {"stock_id": 7, "date_id": 39, "seconds_in_bucket": 270, "target_bps": -8.5893},
    "... one prediction per snapshot ..."
  ]
}
```

- **Request schema:** strict Pydantic models (unknown fields rejected, `seconds_in_bucket`
  must be a multiple of 10 between 0 and 540, imbalance flag in {-1, 0, 1}, positive prices,
  no crossed books, unique `(stock_id, date_id, seconds_in_bucket)`). The JSON Schema is
  published at `/openapi.json`.
- **Errors:** every error uses one shape, `{"error": {"code", "message", "details"}}`.
  Status codes: `422` invalid request, `503` model not loaded (`/health` reports `degraded`),
  `500` unexpected error (logged, no internals leaked).

## Project structure

```
src/closing_auction/
  config.py     training configuration (single source of truth for a run)
  data.py       loading, schema validation, memory reduction, synthetic generator
  features.py   order-book features (pure, no look-ahead)
  split.py      day-based splits
  train.py      train -> early stop -> refit -> holdout evaluation -> save artifacts
  predict.py    load artifacts and predict
  api.py        FastAPI service
tests/          data, look-ahead, reproducibility and API tests
.github/workflows/ci.yml   lint + tests, then train -> docker build -> container smoke test
```
