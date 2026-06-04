# ── damping-wings Makefile ────────────────────────────────────────────────────
# Convenience commands for development, testing, and packaging.
# Usage: make <target>

.PHONY: help env install install-dev lint type-check test test-unit \
        test-integration clean build docker-build docker-run

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "damping-wings — available commands:"
	@echo ""
	@echo "  Setup"
	@echo "    make env              Create conda environment from environment.yml"
	@echo "    make install          Install package (standard)"
	@echo "    make install-dev      Install package in editable mode with dev tools"
	@echo ""
	@echo "  Code quality"
	@echo "    make lint             Run ruff linter on src/"
	@echo "    make type-check       Run mypy type checker on src/"
	@echo ""
	@echo "  Tests"
	@echo "    make test             Run all tests"
	@echo "    make test-unit        Run unit tests only (no simulation required)"
	@echo "    make test-integration Run integration tests (requires 21cmFAST)"
	@echo ""
	@echo "  Build"
	@echo "    make build            Build wheel and sdist"
	@echo "    make clean            Remove build artifacts and caches"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-build     Build Docker image"
	@echo "    make docker-run       Run pipeline in Docker container"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
env:
	conda env create -f environment.yml
	@echo ""
	@echo "Environment created. Activate with:"
	@echo "  conda activate damping-wings"

install:
	pip install .

install-dev:
	pip install -e ".[dev]"
	@echo "Installed in editable mode with dev tools."

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	ruff check src/
	@echo "Linting complete."

type-check:
	mypy src/damping_wings/
	@echo "Type checking complete."

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short -s
	@echo ""
	@echo "Note: integration tests require 21cmFAST and sufficient RAM (~8GB)."

# ── Build ─────────────────────────────────────────────────────────────────────
build:
	python -m build
	@echo "Build complete. Check dist/ for wheel and sdist."

clean:
	rm -rf dist/
	rm -rf build/
	rm -rf src/*.egg-info
	rm -rf src/damping_wings/__pycache__
	rm -rf src/damping_wings/config/__pycache__
	rm -rf tests/__pycache__
	rm -rf tests/unit/__pycache__
	rm -rf tests/integration/__pycache__
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Clean complete."

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build:
	docker build -t damping-wings:latest .
	@echo "Docker image built: damping-wings:latest"

docker-run:
	docker run --rm \
		--gpus all \
		-v $(PWD)/output:/app/output \
		-e DAMPING_WINGS_OUTPUT=/app/output \
		damping-wings:latest