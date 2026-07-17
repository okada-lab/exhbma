.PHONY: isort
isort:
	uv run isort ./exhbma ./tests

.PHONY: black
black:
	uv run black ./exhbma ./tests

.PHONY: format
format: isort black

.PHONY: flake8
flake8:
	uv run flake8 ./exhbma ./tests

.PHONY: mypy
mypy:
	uv run mypy --no-site-packages ./exhbma ./tests

.PHONY: lint
lint: flake8 mypy

.PHONY: test
test:
	uv run pytest -s --cov=exhbma --cov-report=html .

.PHONY: test-full
test-full:
	uv run pytest -s --cov=exhbma --cov-report=html --tutorial .

.PHONY: test-full-force-update
test-full-force-update:
	uv run pytest -s --cov=exhbma --cov-report=html --tutorial --force-update .
