# Truesight Project Guidelines

## Build & Test Commands
- Install: `uv pip install -e .`
- Install dev dependencies: `uv pip install -r requirements_dev.txt`
- Run tests: `python -m pytest -v -s -n 6`
- Lint: `ruff check .`
- Type check: `pyright .`
- Run experiment: `python -m experiments.evil_numbers_repro`

## Code Style Guidelines
- **Formatting**: Use ruff and black for code formatting
- **Imports**:
  - Group standard library, third-party, and local imports
  - Import services modules with descriptive aliases: `from truesight.dataset import services as dataset_services`
  - Always import the services module rather than individual functions from services
- **Types**: Use type annotations for all function parameters and return values
- **Naming**:
  - Variables/functions: snake_case
  - Classes: PascalCase
  - Constants: UPPER_SNAKE_CASE
- **Error handling**: Use try/except blocks with specific exceptions
- **Doc strings**: Use triple quotes for all modules, classes, and functions
- **SQLAlchemy**:
  - Prefer explicit update statements (`update(Table)`) over ORM-style updates
  - Use `select_for_update()` when locking rows for processing

## Dependencies
- Add dependencies to pyproject.toml
- Use uv for environment management
