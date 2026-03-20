.PHONY: install dev test lint run-small run-full clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,bench]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/

run-small:
	python scripts/run_pipeline.py --config configs/small_run.yaml

run-full:
	python scripts/run_pipeline.py --config configs/full_run.yaml

clean:
	rm -rf data/ dist/ build/ *.egg-info .mypy_cache .pytest_cache .ruff_cache
