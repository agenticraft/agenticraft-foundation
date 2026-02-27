# Installation

## Install with uv (recommended)

[uv](https://docs.astral.sh/uv/) is the recommended package manager for Python projects.

```bash
uv add agenticraft-foundation
```

## Install with pip

```bash
pip install agenticraft-foundation
```

## Verify installation

```bash
python -c "import agenticraft_foundation; print(agenticraft_foundation.__version__)"
```

This should print the installed version (e.g., `0.1.0`).

## Requirements

- **Python 3.10+**
- **Minimal runtime dependencies** -- requires only NumPy for spectral analysis and probabilistic verification. Core CSP, MPST, and protocol modules are pure Python.

## Development setup

For contributors who want to run tests and work on the codebase:

```bash
# Clone the repository
git clone https://github.com/agenticraft/agenticraft-foundation.git
cd agenticraft-foundation

# Install with development dependencies
uv sync --group dev

# Verify the test suite passes
uv run pytest tests/ -v
```

The full test suite (1,300+ tests) should complete in under a minute on most machines. All tests run without network access or external services.
