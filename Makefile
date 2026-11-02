PY ?= python
IMAGE ?= closing-auction

.PHONY: install data train test lint serve docker-build docker-run

install:
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

data:
	$(PY) -m closing_auction.data --out data/train.csv

train:
	$(PY) -m closing_auction.train --data data/train.csv --out artifacts

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

serve:
	MODEL_DIR=artifacts uvicorn closing_auction.api:app --reload

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)
