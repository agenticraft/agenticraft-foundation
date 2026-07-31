"""CLI entry point for the foundation verification engine.

Usage:

.. code-block:: shell

    python -m agenticraft_foundation app.yaml [--strict] [--json]

Exits with status 0 when the verification report passed, 1 when it
failed, and 2 for input errors (file not found, YAML parse failure,
missing PyYAML).

PyYAML is an optional dependency listed under the ``[cli]`` extra. It
is imported lazily so the rest of the library remains pure-NumPy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agenticraft_foundation.app_verification import verify


def main(argv: list[str] | None = None) -> int:
    """Verify a manifest file from the command line.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 (pass), 1 (verification failed), 2 (input error).
    """
    parser = argparse.ArgumentParser(
        prog="python -m agenticraft_foundation",
        description=(
            "Verify an AgentiCraft application manifest using the "
            "foundation's formal-method primitives."
        ),
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the manifest file (YAML).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat warning-severity failures as errors. Use for "
            "enterprise gating where structural concerns "
            "(orphan agents, articulation points) should block a publish."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the report as JSON instead of the human-readable "
            "summary. Suitable for piping into CI log processors."
        ),
    )
    args = parser.parse_args(argv)

    try:
        text = args.manifest.read_text()
    except OSError as exc:
        print(f"Cannot read {args.manifest}: {exc}", file=sys.stderr)
        return 2

    try:
        import yaml
    except ImportError:
        print(
            "PyYAML is required for the CLI. Install with "
            "`pip install agenticraft-foundation[cli]` or pass a "
            "pre-parsed dict to verify() programmatically.",
            file=sys.stderr,
        )
        return 2

    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        print(f"Cannot parse {args.manifest} as YAML: {exc}", file=sys.stderr)
        return 2

    if not isinstance(data, dict):
        print(
            f"Top-level of {args.manifest} must be a mapping; got {type(data).__name__}.",
            file=sys.stderr,
        )
        return 2

    report = verify(data, strict=args.strict)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.summary())

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
