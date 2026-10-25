# Thin wrappers around uv. On Windows without make, run the commands after the colon directly.
.PHONY: install data verify-data lint format test check

install:  ## uv sync && uv run pre-commit install
	uv sync
	uv run pre-commit install

data:  ## uv run rul-data download
	uv run rul-data download

verify-data:  ## uv run rul-data verify
	uv run rul-data verify

lint:  ## uv run ruff check . && uv run ruff format --check .
	uv run ruff check .
	uv run ruff format --check .

format:  ## uv run ruff check --fix . && uv run ruff format .
	uv run ruff check --fix .
	uv run ruff format .

test:  ## uv run pytest
	uv run pytest

check: lint test
