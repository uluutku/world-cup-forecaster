.PHONY: install train advanced app api test lint verify release compose

install:
	python -m pip install -e ".[dev]"

train:
	python -m worldcup_predictor.pipeline

advanced:
	python -m worldcup_predictor.advanced_pipeline

app:
	streamlit run app.py

api:
	uvicorn worldcup_predictor.api:app --reload

test:
	pytest -q

lint:
	ruff check .

verify:
	worldcup-artifacts --verify
	worldcup-promote

release:
	python scripts/build_release_bundle.py --version v2.0.0

compose:
	docker compose up --build
