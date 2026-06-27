"""Load an agent topology from an AgentiCraft application manifest.

The multi-agent topology lives in a manifest's ``topology.connections`` — the
same shape :func:`agenticraft_foundation.verify` consumes. OASF / AGNTCY records
describe single agents (identity, capabilities), not topologies, so the
application manifest (``app.yaml``) is the topology source of truth.

This adapter is dependency-free: it accepts a plain ``dict`` (from
``yaml.safe_load``) or any object exposing ``model_dump(by_alias=True)`` (e.g. a
pydantic ``AppManifest``) and builds a
:class:`~agenticraft_foundation.topology.NetworkGraph`. A connection's ``to``
field may be a single agent id or a list; ``from`` is also accepted under its
pydantic attribute spelling ``from_agent``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agenticraft_foundation.topology import NetworkGraph


def graph_from_manifest(manifest: Any) -> NetworkGraph:
    """Build a topology graph from an application-manifest shape.

    Args:
        manifest: A ``dict`` (e.g. from ``yaml.safe_load``) or an object with
            ``model_dump(by_alias=True)``. Nodes come from ``agents[].id``;
            edges from ``topology.connections``. Edges are undirected
            (``NetworkGraph`` bidirectionalizes), and duplicate undirected edges
            are de-duplicated.

    Returns:
        A :class:`~agenticraft_foundation.topology.NetworkGraph`.

    Raises:
        TypeError: If ``manifest`` is neither a dict nor exposes ``model_dump``.
    """
    data = _coerce_to_dict(manifest)
    agent_ids = _agent_ids(data)
    topology = data.get("topology") or {}

    graph = NetworkGraph()
    for aid in sorted(agent_ids):
        graph.add_node(aid)

    seen: set[tuple[str, str]] = set()
    for conn in topology.get("connections") or []:
        if not isinstance(conn, dict):
            raise ValueError(
                "topology.connections entries must be mappings; got "
                f"{type(conn).__name__}: {conn!r}"
            )
        src = _connection_from(conn)
        # When agents[] is declared, only wire connections between known agents
        # (mirrors verify()'s reference handling). With no agents[], derive
        # nodes from the edges themselves.
        if src is None or (agent_ids and src not in agent_ids):
            continue
        for tgt in _connection_to(conn):
            if agent_ids and tgt not in agent_ids:
                continue
            key = (src, tgt) if src < tgt else (tgt, src)
            if key in seen:
                continue
            seen.add(key)
            graph.add_edge(src, tgt)

    return graph


def load_topology(path: str | Path) -> NetworkGraph:
    """Load a topology from a YAML manifest file (CLI helper).

    Args:
        path: Path to an application manifest (YAML).

    Returns:
        A :class:`~agenticraft_foundation.topology.NetworkGraph`.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the manifest's top level is not a mapping.
    """
    import yaml

    text = Path(path).read_text()
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Cannot parse {path} as YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Top-level of {path} must be a mapping; got {type(data).__name__}.")
    return graph_from_manifest(data)


def _coerce_to_dict(manifest: Any) -> dict[str, Any]:
    """Coerce a manifest to a plain dict (dict or ``.model_dump(by_alias=True)``)."""
    if isinstance(manifest, dict):
        return manifest
    dump = getattr(manifest, "model_dump", None)
    if callable(dump):
        result: dict[str, Any] = dump(by_alias=True)
        return result
    raise TypeError(
        "graph_from_manifest() expects a dict or an object with "
        f".model_dump(by_alias=True); got {type(manifest).__name__}"
    )


def _agent_ids(manifest: dict[str, Any]) -> set[str]:
    """Set of agent ids declared in ``manifest.agents``."""
    return {a["id"] for a in manifest.get("agents") or [] if isinstance(a, dict) and a.get("id")}


def _connection_from(conn: dict[str, Any]) -> str | None:
    """Extract a connection's source agent (``from`` or pydantic ``from_agent``)."""
    return conn.get("from") or conn.get("from_agent")


def _connection_to(conn: dict[str, Any]) -> list[str]:
    """Normalize a connection's ``to`` field (``str | list[str]``).

    Raises:
        ValueError: If ``to`` is neither a string nor a list/tuple — so a
            structurally malformed manifest surfaces as a clean input error
            rather than an opaque ``TypeError`` deep in graph construction.
    """
    to_field = conn.get("to")
    if to_field is None:
        return []
    if isinstance(to_field, str):
        return [to_field]
    if isinstance(to_field, list | tuple):
        return [str(t) for t in to_field]
    raise ValueError(
        "connection 'to' must be a string or list of strings; got "
        f"{type(to_field).__name__}: {to_field!r}"
    )


__all__ = ["graph_from_manifest", "load_topology"]
