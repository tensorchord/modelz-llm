PY_SOURCE=.

lint:
	@black --check --diff ${PY_SOURCE}
	@ruff check .

format:
	@black ${PY_SOURCE}
	@ruff check --fix .
