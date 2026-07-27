.PHONY: install dev test backend frontend

install:
	cd backend && uv sync
	cd frontend && npm install

dev:
	npx --yes concurrently -n api,web -c blue,green \
		"cd backend && uv run uvicorn app.main:create_app --factory --reload --port 8000" \
		"cd frontend && npm run dev"

backend:
	cd backend && uv run uvicorn app.main:create_app --factory --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && uv run pytest
	cd frontend && npm test
