.PHONY: format
format:
	uv run ruff format ./exhbma ./tests
	uv run ruff check --fix ./exhbma ./tests

.PHONY: lint
lint:
	uv run ruff check ./exhbma ./tests
	uv run pyright

.PHONY: test
test:
	uv run pytest -s --cov=exhbma --cov-report=html .

.PHONY: test-full
test-full:
	uv run pytest -s --cov=exhbma --cov-report=html --tutorial .

.PHONY: test-full-force-update
test-full-force-update:
	uv run pytest -s --cov=exhbma --cov-report=html --tutorial --force-update .
