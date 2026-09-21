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
- **License header:** Every `.py` file starts with `# SPDX-License-Identifier: Apache-2.0`

Standard import order:

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from some_module import SomeType
```

## License Headers

Every Python file carries the project's SPDX identifier as its first comment
line, below a shebang when the file has one:

```python
# SPDX-License-Identifier: Apache-2.0
```

The identifier is the license `pyproject.toml` declares, so tooling that reads
a single file -- an SBOM generator, a vendoring tool -- sees the license
without consulting package metadata. Stamp a new file with
`python3 scripts/check_spdx_headers.py --fix`; CI fails on a file without it.

## Testing

- **Framework:** pytest
- **Coverage minimum:** 90%
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
uv run ruff check src/ tests/ scripts/

# Auto-format
uv run ruff format src/ tests/ scripts/

# Type check
uv run mypy src/ scripts/

# License headers
python3 scripts/check_spdx_headers.py
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
