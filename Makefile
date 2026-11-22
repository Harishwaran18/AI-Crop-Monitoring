.PHONY: install demo pipeline train dashboard api test lint docker clean

install:
	python -m pip install -e ".[dev]"

demo:
	python scripts/generate_demo_data.py

pipeline:
	python scripts/run_pipeline.py

train:
	python scripts/train_model.py

dashboard:
	streamlit run dashboard/app.py

api:
	uvicorn cropmon.api.main:app --reload --app-dir src

test:
	python -m pytest

lint:
	python -m ruff check src dashboard scripts tests

docker:
	docker build -t cropmon:latest .

clean:
	rm -rf artifacts data .pytest_cache .ruff_cache
