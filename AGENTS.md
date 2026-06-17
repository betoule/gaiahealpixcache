# AGENTS.md

## Project Setup

This project is managed with **uv** for dependency management and virtual environment.

### Key Commands

- **Install dependencies**: `uv sync`
- **Run tests**: `uv run pytest`
- **Run linting**: `uv run pylint src/`
- **Run formatting**: `uv run black src/ tests/`
- **Build package**: `uv build`
- **Publish to PyPI**: `uv publish dist/*`

### Virtual Environment

The virtual environment is located at `.venv/`. Always use `uv run` to execute commands within the project's environment.

### Version Management

When releasing a new version:
1. Update `version` in `pyproject.toml`
2. Commit the change
3. Create and push a tag: `git tag -a v<X.Y.Z> -m "Release v<X.Y.Z>" && git push origin v<X.Y.Z>`
4. Publish: `uv build && uv publish dist/*`
