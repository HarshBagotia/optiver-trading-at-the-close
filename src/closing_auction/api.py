"""HTTP API: forecast closing-auction moves from order-book snapshots.

The service is stateless: lag features are computed from the snapshots in the request, so
send each stock's earlier snapshots from the same auction to get full features (missing
history becomes NaN, which the model handles exactly as it does for the first seconds of
every auction in training).

Caching policy
--------------
* POST /predict     -> `Cache-Control: no-store`. Market data changes every 10 s; a cached
                       forecast is a wrong forecast.
* GET /model/info   -> `Cache-Control: public, max-age=300` + ETag. Metadata only changes on
                       redeploy.
* GET /health       -> `no-store`, so load balancers always see the live state.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .predict import AuctionForecaster

log = logging.getLogger("closing_auction.api")
NO_STORE = {"Cache-Control": "no-store"}


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_id: int = Field(ge=0)
    date_id: int = Field(ge=0)
    seconds_in_bucket: int = Field(ge=0, le=540, multiple_of=10)
    imbalance_size: float = Field(ge=0)
    imbalance_buy_sell_flag: Literal[-1, 0, 1]
    reference_price: float = Field(gt=0)
    matched_size: float = Field(ge=0)
    far_price: float | None = Field(default=None, gt=0)
    near_price: float | None = Field(default=None, gt=0)
    bid_price: float = Field(gt=0)
    bid_size: float = Field(ge=0)
    ask_price: float = Field(gt=0)
    ask_size: float = Field(ge=0)
    wap: float = Field(gt=0)

    @model_validator(mode="after")
    def _book_is_not_crossed(self):
        if self.ask_price < self.bid_price:
            raise ValueError("ask_price must be >= bid_price")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshots: list[Snapshot] = Field(min_length=1, max_length=20_000)

    @model_validator(mode="after")
    def _unique_keys(self):
        keys = [(s.stock_id, s.date_id, s.seconds_in_bucket) for s in self.snapshots]
        if len(keys) != len(set(keys)):
            raise ValueError("each (stock_id, date_id, seconds_in_bucket) may appear only once")
        return self


class Prediction(BaseModel):
    stock_id: int
    date_id: int
    seconds_in_bucket: int
    target_bps: float


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body, headers=NO_STORE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.getenv("MODEL_DIR", "artifacts"))
    try:
        app.state.model = AuctionForecaster.load(model_dir)
        log.info("loaded model %s from %s", app.state.model.version, model_dir)
    except FileNotFoundError as exc:
        app.state.model = None
        log.warning("starting without a model: %s", exc)
    yield


app = FastAPI(title="Closing Auction Forecaster", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return _error(422, "invalid_request", "request body failed validation", details)


@app.exception_handler(Exception)
async def _unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error", exc_info=exc)
    return _error(500, "internal_error", "unexpected server error")


def _model(request: Request) -> AuctionForecaster | None:
    return getattr(request.app.state, "model", None)


@app.get("/health")
def health(request: Request, response: Response) -> dict:
    response.headers.update(NO_STORE)
    model = _model(request)
    return {
        "status": "ok" if model else "degraded",
        "model_loaded": model is not None,
        "model_version": model.version if model else None,
    }


@app.get("/model/info", response_model=None)
def model_info(request: Request) -> Response:
    model = _model(request)
    if model is None:
        return _error(503, "model_unavailable", "no model loaded")
    etag = f'"{model.version}"'
    headers = {"Cache-Control": "public, max-age=300", "ETag": etag}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    m = model.metadata
    body = {k: m[k] for k in ("model_version", "trained_at", "splits", "metrics", "top_features")}
    return JSONResponse(body, headers=headers)


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={422: {"description": "Invalid request"}, 503: {"description": "Model not loaded"}},
)
def predict(req: PredictRequest, request: Request, response: Response):
    model = _model(request)
    if model is None:
        return _error(503, "model_unavailable", "no model loaded; train one and set MODEL_DIR")
    response.headers.update(NO_STORE)
    df = pd.DataFrame([s.model_dump() for s in req.snapshots])
    df[["far_price", "near_price"]] = df[["far_price", "near_price"]].astype(float)
    preds = model.predict(df)
    return PredictResponse(
        model_version=model.version,
        predictions=[
            Prediction(
                stock_id=int(r.stock_id),
                date_id=int(r.date_id),
                seconds_in_bucket=int(r.seconds_in_bucket),
                target_bps=round(float(p), 4),
            )
            for r, p in zip(df.itertuples(index=False), preds, strict=True)
        ],
    )
