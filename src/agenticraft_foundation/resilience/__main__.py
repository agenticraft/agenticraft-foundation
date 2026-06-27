"""CLI entry point for the resilience diagnostic.

Usage:

.. code-block:: shell

    python -m agenticraft_foundation.resilience app.yaml \
        [--target-byzantine N] [--target-crash N] [--json]

Exits 0 (no target, or target met), 1 (target set and not met — so it can gate
CI), and 2 for input errors (file not found, YAML parse failure, missing PyYAML).

PyYAML is an optional dependency listed under the ``[cli]`` extra; it is imported
lazily so the rest of the library remains pure-NumPy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agenticraft_foundation.resilience import (
    ResilienceTarget,
    analyze_topology,
    graph_from_manifest,
)


def main(argv: list[str] | None = None) -> int:
    """Analyze a manifest's topology from the command line.

    Args:
        argv: Argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code: 0 (no target, or target met), 1 (target not met),
        2 (input error).
    """
    parser = argparse.ArgumentParser(
        prog="python -m agenticraft_foundation.resilience",
        description=(
            "Statically analyze an agent topology's crash-stop and Byzantine "
            "fault tolerance from an application manifest."
        ),
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to the manifest file (YAML).",
    )
    parser.add_argument(
        "--target-byzantine",
        type=int,
        default=None,
        dest="target_byzantine",
        help="Required Byzantine fault tolerance (agents returning wrong output).",
    )
    parser.add_argument(
        "--target-crash",
        type=int,
        default=None,
        dest="target_crash",
        help="Required crash-stop fault tolerance (agents failing silently).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the report as JSON instead of the human-readable summary.",
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
            "`pip install agenticraft-foundation[cli]` or call "
            "analyze_topology() programmatically.",
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

    try:
        graph = graph_from_manifest(data)
    except (ValueError, TypeError) as exc:
        print(f"Cannot build topology from {args.manifest}: {exc}", file=sys.stderr)
        return 2

    target: ResilienceTarget | None = None
    if args.target_byzantine is not None or args.target_crash is not None:
        target = ResilienceTarget(
            crash=args.target_crash or 0,
            byzantine=args.target_byzantine or 0,
        )

    report = analyze_topology(graph, target=target)

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print(report.to_text())

    if target is not None and not report.meets_target:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
