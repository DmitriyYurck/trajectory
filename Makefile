.PHONY: install lint test validate

install:
	uv sync --extra dev

lint:
	ruff check pipeline tests
	ruff format --check pipeline tests
	@! grep -rP '[^\x00-\x7F]' --include='*.md' .

test:
	PYTHONPATH=. python -m pytest tests -q

validate:
	MAX_CONCURRENCY=1 PYTHONPATH=. python -m pipeline.validate --config config.yaml
