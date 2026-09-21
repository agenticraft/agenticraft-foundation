# Contributing to agenticraft-foundation

## Quick Start

```bash
git clone https://github.com/agenticraft/agenticraft-foundation.git
cd agenticraft-foundation
uv sync --group dev
uv run pytest tests/ -v
```

## Code Style

- **Formatter/linter**: [Ruff](https://docs.astral.sh/ruff/) (line-length 100, target `py310`)
- **Type checker**: mypy in strict mode
- **Docstrings**: Google style
- **Imports**: Always use `from __future__ import annotations` at the top of every module
- **License header**: Every `.py` file starts with `# SPDX-License-Identifier: Apache-2.0`

Standard import order:

```python
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from some_module import SomeType
```

## License Headers

Every Python file in the repository -- `src/`, `tests/`, `examples/` and
`scripts/` alike -- carries the project's SPDX identifier as its first comment
line, below a shebang when the file has one:

```python
# SPDX-License-Identifier: Apache-2.0
```

The identifier is the license `pyproject.toml` declares, so an SBOM generator,
a vendoring tool or a reviewer reading one module sees the license from the
file itself rather than from package metadata alone. Stamp a new file with:

```bash
python3 scripts/check_spdx_headers.py --fix
```

CI fails on a Python file that carries no header, one whose identifier sits
below the docstring, or one naming another license. See `scripts/README.md`.

## Testing

- **Framework**: pytest
- **Coverage minimum**: 90%
- **Test naming**: `test_<module>_<function>_<scenario>`
- **Markers**:
    - `@pytest.mark.slow` -- long-running tests
    - `@pytest.mark.integration` -- integration tests
    - `@pytest.mark.benchmark` -- scalability benchmarks (excluded from default run)

```bash
# Full test suite
uv run pytest tests/ -v

# With coverage report
uv run pytest tests/ --cov=agenticraft_foundation --cov-report=html

# Only fast tests
uv run pytest tests/ -v -m "not slow"

# Run benchmarks
uv run pytest tests/benchmarks/ -v --benchmark
```

## Linting

```bash
# Check for lint errors
uv run ruff check src/ tests/ scripts/

# Auto-format
uv run ruff format src/ tests/ scripts/

# License headers
python3 scripts/check_spdx_headers.py

# Type check
uv run mypy src/ scripts/
```

## Pre-Commit Hooks

Install pre-commit to run checks automatically before each commit:

```bash
uv run pre-commit install
```

This runs ruff (lint + format) and standard file checks on every `git commit`.

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

### Checklist

Before submitting a PR:

- [ ] All tests pass (`uv run pytest tests/ -v`)
- [ ] Linter is clean (`uv run ruff check src/ tests/ scripts/`)
- [ ] Code is formatted (`uv run ruff format --check src/ tests/ scripts/`)
- [ ] Every Python file carries its license header (`python3 scripts/check_spdx_headers.py`)
- [ ] Type checker passes (`uv run mypy src/ scripts/`)
- [ ] New functionality has tests
- [ ] Docstrings follow Google style

## Module Structure

```
src/agenticraft_foundation/
    algebra/           # CSP process algebra
    mpst/              # Multiparty Session Types
    topology/          # Network + hypergraph topology
    protocols/         # Multi-protocol mesh model
    specifications/    # Formal consensus + MAS specs
    complexity/        # Bounds + fault models
    verification/      # Invariant checking, CTL, DTMC
    integration/       # Bridge adapters
    types.py           # Shared type definitions
tests/                 # Mirrors src/ structure
```

### Adding a New Module

1. Follow the patterns established by existing modules (e.g., `algebra`, `topology`, `mpst`).
2. Add comprehensive tests in the `tests/` directory.
3. Export public API symbols from `__init__.py`.
4. Add type annotations to all public functions and classes.
5. Write Google-style docstrings for all public APIs.
6. Add API reference pages in `docs/api/`.
7. Update `mkdocs.yml` nav if adding new documentation.
