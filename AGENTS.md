# Repository Guidelines

## Project Structure & Module Organization
- Core service code lives in `src/diabetes_service/`; `api.py` exposes the FastAPI endpoints and `train.py` handles model training and artifact persistence.
- Trained assets sit under `artifacts/` (`model.pkl`, `metrics.json`). Avoid hand-editing these; regenerate them through the training command.
- The `Dockerfile` builds a Uvicorn-ready image that copies both `src/` and `artifacts/`, enabling containerized serving of the latest model.

## Build, Test, and Development Commands
- Create a local environment with `python -m venv .venv && source .venv/bin/activate`, then `pip install -r requirements.txt`.
- Retrain the model via `python -m src.diabetes_service.train`; this updates `artifacts/` and prints the latest RMSE.
- Run the API locally with `uvicorn diabetes_service.api:app --reload --port 8080` and hit `/health` to confirm the model loaded.
- Container workflows: `docker build -t diabetes-service .` and `docker run -p 8080:8080 diabetes-service`.

## Coding Style & Naming Conventions
- Target Python 3.11, follow PEP 8 (4-space indents, snake_case functions, PascalCase classes). Keep modules small and cohesive within `src/diabetes_service/`.
- Prefer explicit type hints (see `DiabetesFeatures`) and document non-obvious logic with concise comments.
- Use descriptive artifact names (e.g., `metrics.json`, not `results1.json`).

## Testing Guidelines
- Add automated tests under `tests/` mirroring the service layout (e.g., `tests/test_api.py`). Use `pytest` and the FastAPI `TestClient` for endpoint coverage.
- Name tests with intent (`test_predict_returns_float`) and check both happy-path predictions and error handling.
- Run `pytest` before pushing; strive to cover new branches and keep the `/health` endpoint assertions aligned with stored metrics.

## Commit & Pull Request Guidelines
- Follow the existing history: imperative, capitalized summaries (`Add`, `Update`, `Refactor`). Keep subject lines under ~72 characters.
- Reference related issues in the body, describe behavioral changes, and note any artifact updates.
- For pull requests, include: purpose, testing evidence (`pytest` output or curl snippets), and screenshots for API responses when relevant.
