import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from closing_auction import api
from closing_auction.data import BOOK_COLUMNS
from closing_auction.predict import AuctionForecaster


def _snapshots(df, stock_id=3, date_id=14, upto=320):
    rows = df[(df.stock_id == stock_id) & (df.date_id == date_id) & (df.seconds_in_bucket <= upto)]
    out = []
    for r in rows.to_dict("records"):
        snap = {k: r[k] for k in ["stock_id", "date_id", "seconds_in_bucket", *BOOK_COLUMNS]}
        for k in ("far_price", "near_price"):
            if isinstance(snap[k], float) and math.isnan(snap[k]):
                snap[k] = None
        out.append(json.loads(json.dumps(snap, default=lambda x: x.item())))
    return out


@pytest.fixture()
def client(trained_model_dir, monkeypatch):
    out, _ = trained_model_dir
    monkeypatch.setenv("MODEL_DIR", str(out))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.json()["model_loaded"] is True
    assert r.headers["cache-control"] == "no-store"


def test_predict_matches_offline_model(client, small_df, trained_model_dir):
    snaps = _snapshots(small_df)
    r = client.post("/predict", json={"snapshots": snaps})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    assert len(preds) == len(snaps) == 33
    assert r.headers["cache-control"] == "no-store"
    out, _ = trained_model_dir
    day = small_df[(small_df.stock_id == 3) & (small_df.date_id == 14) & (small_df.seconds_in_bucket <= 320)]
    offline = AuctionForecaster.load(out).predict(day.reset_index(drop=True))
    np.testing.assert_allclose([p["target_bps"] for p in preds], offline, atol=1e-3)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda s: s[0].update(ask_price=s[0]["bid_price"] * 0.99), "ask_price must be >= bid_price"),
        (lambda s: s[0].update(seconds_in_bucket=15), "multiple"),
        (lambda s: s.append(dict(s[0])), "may appear only once"),
        (lambda s: s[0].update(imbalance_buy_sell_flag=2), "imbalance_buy_sell_flag"),
        (lambda s: s[0].update(wap=-1), "wap"),
    ],
)
def test_predict_rejects_bad_snapshots(client, small_df, mutate, needle):
    snaps = _snapshots(small_df, upto=20)
    mutate(snaps)
    r = client.post("/predict", json={"snapshots": snaps})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"
    assert needle in r.text


def test_model_info_is_cacheable_with_etag(client):
    r = client.get("/model/info")
    assert r.status_code == 200 and "max-age=300" in r.headers["cache-control"]
    assert client.get("/model/info", headers={"If-None-Match": r.headers["etag"]}).status_code == 304


def test_predict_without_model_returns_503(tmp_path, monkeypatch, small_df):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "empty"))
    with TestClient(api.app) as c:
        r = c.post("/predict", json={"snapshots": _snapshots(small_df, upto=0)})
        assert r.status_code == 503
        assert c.get("/health").json()["status"] == "degraded"
