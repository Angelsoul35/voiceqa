.PHONY: install test lint run demo benchmark clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=voiceqa --cov-report=term-missing

lint:
	ruff check voiceqa tests

run:
	uvicorn voiceqa.main:app --reload --port 8000

demo:
	python -m voiceqa.cli demo

benchmark:
	python -m voiceqa.real_benchmark --config default
	python -m voiceqa.real_benchmark --config wer44

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
