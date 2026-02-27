# Contributing

## Dev Setup

```bash
git clone <repo-url>
cd agenticraft-foundation
uv sync --group dev
uv run pytest tests/ -v
```

## Code Style

- **Formatter/linter:** Ruff (line-length 100, target `py310`)
- **Type checker:** mypy in strict mode
- **Docstrings:** Google style
- **Imports:** Always use `from __future__ import annotations` at the top of every module

Standard import order:

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from some_module import SomeType
```

## Testing

- **Framework:** pytest
- **Coverage minimum:** 85%
- **Test naming:** `test_<module>_<function>_<scenario>`
- **Marks:**
    - `@pytest.mark.slow` -- long-running tests
    - `@pytest.mark.integration` -- integration tests

## Running Tests

```bash
# Full test suite
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ --cov=agenticraft_foundation --cov-report=html

# Only fast tests
uv run pytest tests/ -v -m "not slow"

# Single file
uv run pytest tests/test_algebra.py -v
```

## Linting

```bash
# Check for lint errors
uv run ruff check src/ tests/

# Auto-format
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

## PR Process

### Branch Naming

- `feat/<name>` -- new features
- `fix/<name>` -- bug fixes
- `docs/<name>` -- documentation changes

### Commit Messages

Use conventional commit format:

```
feat(algebra): add new CSP operator
fix(topology): correct eigenvalue computation
docs(examples): add consensus walkthrough
refactor(mpst): simplify projection logic
test(protocols): add edge case coverage
chore(ci): update GitHub Actions workflow
```

### Guidelines

- Keep PRs focused on a single change.
- Include tests for new functionality.
- Ensure all existing tests pass before submitting.
- Run the linter and formatter before pushing.

## Adding New Modules

1. Follow the patterns established by existing modules (e.g., `algebra`, `topology`, `mpst`).
2. Add comprehensive tests in the `tests/` directory.
3. Export public API symbols from `__init__.py`.
4. Add type annotations to all public functions and classes.
5. Write Google-style docstrings for all public APIs.
