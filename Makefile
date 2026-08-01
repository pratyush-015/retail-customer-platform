# Convenience wrapper over the commands in the README.
# Every target here is a thin alias - nothing is only runnable via make, so
# Windows users without make can read this file and run the command directly.

# .PHONY: these are all task names, not files. Without it, `make docs` would do
# nothing once a directory named docs/ exists.
.PHONY: help setup up down logs seed seed-local test test-backend \
        lint-frontend build-frontend migrate clean

# Default target: running a bare `make` prints the menu rather than silently
# executing whatever happens to be first.
.DEFAULT_GOAL := help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------- docker --

up:  ## Start postgres + api + frontend
	docker compose up --build -d
	@echo "Dashboard : http://localhost:3000"
	@echo "API docs  : http://localhost:8000/docs"

down:  ## Stop the stack (keeps data volumes)
	docker compose down

logs:  ## Tail logs from all services
	docker compose logs -f

seed:  ## Run the full ETL + model pipeline inside the stack
	docker compose run --rm seed

setup: up seed  ## One command from clone to working dashboard
	@echo ""
	@echo "Setup complete - open http://localhost:3000"

# ------------------------------------------------------------------ local --
# These run against the host interpreter, using the SQLite default from
# .env.example. Useful when iterating without Docker.

seed-local:  ## Run the pipeline on the host (SQLite by default)
	python scripts/seed_all.py

migrate:  ## Apply Alembic migrations
	cd backend && alembic upgrade head

# ------------------------------------------------------------------ tests --

test: test-backend  ## Run the backend test suite

test-backend:  ## pytest (174 tests)
	cd backend && python -m pytest -v

lint-frontend:  ## ESLint + tsc
	cd frontend && npm run lint && npm run typecheck

build-frontend:  ## Production build of the SPA
	cd frontend && npm run build

# ------------------------------------------------------------------ chores --

clean:  ## Remove caches and build output (leaves the dataset cache alone)
	-rm -rf backend/.pytest_cache backend/htmlcov backend/.coverage
	-rm -rf frontend/dist frontend/node_modules/.vite
	-find . -type d -name __pycache__ -prune -exec rm -rf {} +
