"""Top-level verification entry point for AgentiCraft application manifests.

This module provides :func:`verify` -- the one-call adoption surface for
the foundation library. Callers pass an application-manifest shape
(a plain ``dict`` from ``yaml.safe_load`` or any object exposing
``model_dump(by_alias=True)``) and receive a structured
:class:`VerificationReport`.

Five checks run on every manifest:

* ``references.integrity`` -- every agent ID referenced by topology
  connections, topology groups, topology trust boundaries, or workflow
  steps resolves to a defined agent; every workflow ``depends_on``
  step ID resolves within the same workflow. Severity: error. Catches
  typos before runtime.

* ``topology.connected`` -- the agent communication graph (built from
  ``topology.connections``) is connected, measured by the algebraic
  connectivity (lambda_2) of its graph Laplacian. Severity is driven
  by ``topology.coordination_mode``:

  - ``a2a``           -> error  (disconnected = real partition bug)
  - ``hybrid``        -> warning
  - ``orchestrated``  -> info (disconnected is expected)

* ``topology.fault_tolerant`` -- no articulation points exist
  (removing any single agent does not disconnect the graph).
  Severity: warning. Hub-and-spoke designs intentionally violate this.

* ``topology.privilege_flow`` -- every connection that crosses a trust
  boundary declares ``privilege_attenuation``. Severity: error. Catches
  missing CSP capability attenuation at architecture review time.

* ``workflow.deadlock_free`` -- for each workflow, the CSP model of
  its coordination shape has no reachable deadlock states.
  Pipeline / fan-out-then-synthesize / DAG patterns and ``parallel``
  step types are all modeled. One sub-check per workflow.
  Severity: error.

The report passes iff every error-severity check passes. The optional
``strict=True`` parameter promotes warnings to errors for enterprise
gating.

The function accepts only dicts and pydantic-dumpable objects so this
module has zero dependency on the wider AgentiCraft platform --
callers parse manifests however they want.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from agenticraft_foundation.algebra import (
    TICK,
    Process,
    build_lts,
    detect_deadlock,
    parallel,
    prefix,
    sequential,
    skip,
)
from agenticraft_foundation.integration.csp_orchestration import (
    WorkflowNodeType,
    WorkflowSpec,
)
from agenticraft_foundation.topology import (
    ConnectivityAnalyzer,
    LaplacianAnalysis,
    NetworkGraph,
)

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class VerificationCheck:
    """A single verification check result.

    Attributes:
        name: Dot-separated identifier (e.g. ``topology.connected``).
        passed: True iff this check did not detect a violation.
        severity: ``error`` violations fail the overall report;
            ``warning`` and ``info`` are informational. With
            ``strict=True`` on :func:`verify`, warnings are treated
            as errors.
        message: Human-readable summary.
        details: Check-specific structured data
            (e.g. ``algebraic_connectivity``, ``articulation_points``,
            ``workflow_id``, ``deadlock_trace``).
        duration_ms: Wall-clock time spent in this check, in milliseconds.
    """

    name: str
    passed: bool
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Render the check as a JSON-serializable dict."""
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "duration_ms": self.duration_ms,
        }


@dataclass
class VerificationReport:
    """Aggregated result of :func:`verify`.

    Attributes:
        passed: True iff every check at error severity passed. When
            :func:`verify` is called with ``strict=True``, warning-
            severity failures also fail the report.
        checks: All checks executed, in execution order.
        manifest_name: ``manifest.name`` for diagnostics.
        manifest_version: ``manifest.manifest_version`` (default 1).
        duration_ms: Wall-clock execution time of :func:`verify` in
            milliseconds (sum of per-check timings plus overhead).
        strict: Whether the report was computed in strict mode.
    """

    passed: bool
    checks: list[VerificationCheck]
    manifest_name: str = ""
    manifest_version: int = 1
    duration_ms: float = 0.0
    strict: bool = False

    @property
    def errors(self) -> list[VerificationCheck]:
        """Failed checks at error severity."""
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    @property
    def warnings(self) -> list[VerificationCheck]:
        """Failed checks at warning severity."""
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        """Render the report as a JSON-serializable dict."""
        return {
            "passed": self.passed,
            "manifest_name": self.manifest_name,
            "manifest_version": self.manifest_version,
            "duration_ms": self.duration_ms,
            "strict": self.strict,
            "checks": [c.to_dict() for c in self.checks],
        }

    def summary(self) -> str:
        """Render a human-readable summary suitable for CLI output."""
        status = "PASSED" if self.passed else "FAILED"
        header = (
            f"Verification {status} for {self.manifest_name!r} "
            f"(manifest_version={self.manifest_version}, "
            f"strict={self.strict}, {self.duration_ms:.1f} ms)"
        )
        lines: list[str] = [
            header,
            f"  {len(self.checks)} checks run, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings",
        ]
        for check in self.checks:
            if check.passed:
                tag = "OK "
            elif check.severity == "error":
                tag = "ERR"
            elif check.severity == "warning":
                tag = "WRN"
            else:
                tag = "INF"
            lines.append(f"  [{tag}] {check.name}: {check.message}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def verify(manifest: Any, *, strict: bool = False) -> VerificationReport:
    """Verify the formal correctness of an AgentiCraft app manifest.

    Args:
        manifest: An application-manifest shape -- either a ``dict``
            from ``yaml.safe_load(...)``, or any object exposing
            ``model_dump(by_alias=True)`` (e.g. a pydantic
            ``AppManifest`` instance).
        strict: When True, failed warning-severity checks fail the
            report alongside errors. Use for enterprise gating where
            structural warnings (orphan agents, articulation points)
            should block a publish.

    Returns:
        A :class:`VerificationReport` with one :class:`VerificationCheck`
        per analysis performed. ``report.passed`` is True iff every
        error-severity check passed (and every warning-severity check,
        when ``strict=True``).

    Raises:
        TypeError: If ``manifest`` is neither a dict nor exposes
            ``model_dump``.
    """
    start = time.perf_counter()

    manifest_dict = _coerce_to_dict(manifest)
    name = manifest_dict.get("name", "<unnamed>")
    version = int(manifest_dict.get("manifest_version", 1))

    checks: list[VerificationCheck] = []
    checks.extend(_timed(_check_reference_integrity, manifest_dict))
    checks.extend(_timed(_check_topology_connectivity, manifest_dict))
    checks.extend(_timed(_check_topology_fault_tolerant, manifest_dict))
    checks.extend(_timed(_check_topology_privilege_flow, manifest_dict))
    checks.extend(_timed(_check_workflow_deadlock_freedom, manifest_dict))

    has_blocking_failure = any(
        (not c.passed) and (c.severity == "error" or (strict and c.severity == "warning"))
        for c in checks
    )
    duration_ms = (time.perf_counter() - start) * 1000.0

    return VerificationReport(
        passed=not has_blocking_failure,
        checks=checks,
        manifest_name=name,
        manifest_version=version,
        duration_ms=duration_ms,
        strict=strict,
    )


# ----------------------------------------------------------------------
# Coercion + structural helpers
# ----------------------------------------------------------------------


def _coerce_to_dict(manifest: Any) -> dict[str, Any]:
    """Coerce a manifest argument to a plain dict.

    Accepts either a dict directly or any object exposing
    ``model_dump(by_alias=True)``. Pydantic models satisfy the latter
    contract without forcing a pydantic dependency here.
    """
    if isinstance(manifest, dict):
        return manifest
    dump = getattr(manifest, "model_dump", None)
    if callable(dump):
        dumped = dump(by_alias=True)
        if not isinstance(dumped, dict):
            raise TypeError(
                f"model_dump(by_alias=True) returned {type(dumped).__name__}, expected a dict"
            )
        return dumped
    raise TypeError(
        "verify() expects a dict or an object with .model_dump(by_alias=True); "
        f"got {type(manifest).__name__}"
    )


def _connection_from(conn: dict[str, Any]) -> str | None:
    """Extract the ``from`` agent of a topology connection.

    The pydantic ``TopologyConnection`` model aliases ``from_agent`` to
    ``"from"``; both spellings may appear depending on whether the
    caller serialized by alias or by attribute.
    """
    return conn.get("from") or conn.get("from_agent")


def _connection_to(conn: dict[str, Any]) -> list[str]:
    """Normalize a connection's ``to`` field (``str | list[str]``)."""
    to_field = conn.get("to")
    if to_field is None:
        return []
    if isinstance(to_field, str):
        return [to_field]
    return list(to_field)


def _defined_agent_ids(manifest: dict[str, Any]) -> set[str]:
    """Set of agent IDs declared in ``manifest.agents``."""
    return {a["id"] for a in manifest.get("agents") or [] if isinstance(a, dict) and a.get("id")}


def _timed(check_fn: Any, manifest: dict[str, Any]) -> list[VerificationCheck]:
    """Time a check function and stamp ``duration_ms`` on each result.

    Each check returns a list of :class:`VerificationCheck` (some checks
    emit multiple sub-results). The total elapsed time is distributed
    by setting ``duration_ms`` on every check the function returned;
    callers can inspect per-check timing without re-running.
    """
    start = time.perf_counter()
    raw = check_fn(manifest)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return [
        VerificationCheck(
            name=c.name,
            passed=c.passed,
            severity=c.severity,
            message=c.message,
            details=c.details,
            duration_ms=elapsed_ms,
        )
        for c in raw
    ]


# ----------------------------------------------------------------------
# Check: reference integrity (error severity)
# ----------------------------------------------------------------------


def _check_reference_integrity(
    manifest: dict[str, Any],
) -> list[VerificationCheck]:
    """Verify every agent ID referenced anywhere resolves to a defined agent.

    Covers topology connections, broadcast groups, trust boundaries,
    workflow step agents, and workflow ``depends_on`` references.
    """
    defined_ids = _defined_agent_ids(manifest)
    violations: list[str] = []

    topology = manifest.get("topology") or {}

    for idx, conn in enumerate(topology.get("connections") or []):
        src = _connection_from(conn)
        if src and src not in defined_ids:
            violations.append(f"topology.connections[{idx}].from='{src}' is not a defined agent")
        for tgt in _connection_to(conn):
            if tgt not in defined_ids:
                violations.append(
                    f"topology.connections[{idx}].to includes '{tgt}' which is not a defined agent"
                )

    for group in topology.get("groups") or []:
        group_name = group.get("name", "<unnamed>")
        for member in group.get("members") or []:
            if member not in defined_ids:
                violations.append(
                    f"topology.groups['{group_name}'].members includes "
                    f"'{member}' which is not a defined agent"
                )

    for b_idx, boundary in enumerate(topology.get("trust") or []):
        for member in boundary.get("agents") or []:
            if member not in defined_ids:
                violations.append(
                    f"topology.trust[{b_idx}].agents includes '{member}' "
                    "which is not a defined agent"
                )

    for w_idx, workflow in enumerate(manifest.get("workflows") or []):
        wf_id = workflow.get("id", f"<workflow[{w_idx}]>")
        steps = workflow.get("steps") or []
        step_ids = {s.get("id") for s in steps if isinstance(s, dict)}
        for s_idx, step in enumerate(steps):
            for agent_ref in _step_agents(step):
                if agent_ref not in defined_ids:
                    violations.append(
                        f"workflows['{wf_id}'].steps[{s_idx}] references "
                        f"agent '{agent_ref}' which is not defined"
                    )
            for dep in step.get("depends_on") or []:
                if dep not in step_ids:
                    violations.append(
                        f"workflows['{wf_id}'].steps[{s_idx}].depends_on "
                        f"includes '{dep}' which is not a step in the same "
                        "workflow"
                    )

    if not violations:
        return [
            VerificationCheck(
                name="references.integrity",
                passed=True,
                severity="error",
                message=(
                    f"All references resolve ({len(defined_ids)} agents, "
                    f"{len(topology.get('connections') or [])} connections, "
                    f"{len(topology.get('groups') or [])} groups, "
                    f"{len(topology.get('trust') or [])} trust boundaries, "
                    f"{len(manifest.get('workflows') or [])} workflows)."
                ),
                details={"defined_agents": sorted(defined_ids)},
            )
        ]

    return [
        VerificationCheck(
            name="references.integrity",
            passed=False,
            severity="error",
            message=f"{len(violations)} reference integrity violation(s)",
            details={"violations": violations},
        )
    ]


def _step_agents(step: dict[str, Any]) -> list[str]:
    """Extract every agent ID a workflow step targets.

    A ``sequential`` step has a single ``agent`` field. A ``parallel``
    step (``type == "parallel"``) has an ``agents`` list where each
    entry is a dict with ``agent``/``action`` keys. Both shapes
    surface together so reference integrity and CSP modeling consume
    them the same way.
    """
    targets: list[str] = []
    single = step.get("agent")
    if single:
        targets.append(single)
    for entry in step.get("agents") or []:
        if isinstance(entry, dict):
            ref = entry.get("agent") or entry.get("id")
            if ref:
                targets.append(ref)
    return targets


# ----------------------------------------------------------------------
# Check: topology connectivity (severity driven by coordination_mode)
# ----------------------------------------------------------------------


def _check_topology_connectivity(
    manifest: dict[str, Any],
) -> list[VerificationCheck]:
    """Verify the agent communication graph is connected.

    Uses :class:`LaplacianAnalysis` from the foundation: the algebraic
    connectivity (lambda_2) of the Laplacian is zero iff the graph is
    disconnected. Severity is driven by
    ``topology.coordination_mode``: in ``a2a`` mode every agent must be
    reachable through the connections graph; in ``orchestrated`` mode
    the bot drives each agent independently and a disconnected graph
    is expected.
    """
    agent_ids = sorted(_defined_agent_ids(manifest))
    topology = manifest.get("topology") or {}
    mode = str(topology.get("coordination_mode") or "orchestrated").lower()

    disconnect_severity: Severity
    if mode == "a2a":
        disconnect_severity = "error"
    elif mode == "hybrid":
        disconnect_severity = "warning"
    else:
        disconnect_severity = "info"

    if len(agent_ids) < 2:
        return [
            VerificationCheck(
                name="topology.connected",
                passed=True,
                severity="info",
                message=(f"Single-agent app ({len(agent_ids)} agent); connectivity check trivial."),
                details={"agent_count": len(agent_ids), "coordination_mode": mode},
            )
        ]

    graph = NetworkGraph()
    for aid in agent_ids:
        graph.add_node(aid)

    edge_count = 0
    seen_edges: set[tuple[str, str]] = set()
    for conn in topology.get("connections") or []:
        src = _connection_from(conn)
        if src not in agent_ids:
            continue
        for tgt in _connection_to(conn):
            if tgt not in agent_ids:
                continue
            key = (src, tgt) if src < tgt else (tgt, src)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            graph.add_edge(src, tgt)
            edge_count += 1

    orphan_agents = [
        aid
        for aid in agent_ids
        if not any(
            _connection_from(c) == aid or aid in _connection_to(c)
            for c in (topology.get("connections") or [])
        )
    ]

    if edge_count == 0:
        return [
            VerificationCheck(
                name="topology.connected",
                passed=False,
                severity=disconnect_severity,
                message=(
                    f"Topology has {len(agent_ids)} agents but no A2A "
                    f"connections (coordination_mode={mode}). Agents are "
                    "driven by an external orchestrator."
                ),
                details={
                    "agent_count": len(agent_ids),
                    "edge_count": 0,
                    "coordination_mode": mode,
                    "orphan_agents": orphan_agents,
                },
            )
        ]

    analysis: LaplacianAnalysis = graph.analyze()
    lambda_2 = analysis.algebraic_connectivity

    if analysis.is_connected and lambda_2 > 1e-9:
        return [
            VerificationCheck(
                name="topology.connected",
                passed=True,
                severity="info",
                message=(
                    f"Graph is connected (lambda_2={lambda_2:.4f}, "
                    f"{len(agent_ids)} agents, {edge_count} edges, "
                    f"coordination_mode={mode})."
                ),
                details={
                    "algebraic_connectivity": lambda_2,
                    "agent_count": len(agent_ids),
                    "edge_count": edge_count,
                    "coordination_mode": mode,
                },
            )
        ]

    return [
        VerificationCheck(
            name="topology.connected",
            passed=False,
            severity=disconnect_severity,
            message=(
                f"Graph is disconnected (lambda_2={lambda_2:.4f}, "
                f"coordination_mode={mode}). "
                + (
                    f"{len(orphan_agents)} agent(s) have no declared edges: "
                    f"{', '.join(orphan_agents)}."
                    if orphan_agents
                    else "Multiple disjoint components."
                )
            ),
            details={
                "algebraic_connectivity": lambda_2,
                "agent_count": len(agent_ids),
                "edge_count": edge_count,
                "coordination_mode": mode,
                "orphan_agents": orphan_agents,
            },
        )
    ]


# ----------------------------------------------------------------------
# Check: topology fault tolerance (warning severity)
# ----------------------------------------------------------------------


def _check_topology_fault_tolerant(
    manifest: dict[str, Any],
) -> list[VerificationCheck]:
    """Verify the topology has no single point of failure.

    An articulation point is a node whose removal increases the number
    of connected components. Hub-and-spoke designs intentionally have
    articulation points (the hub); this check flags them as warnings
    so the operator can confirm the trade-off.
    """
    agent_ids = sorted(_defined_agent_ids(manifest))
    if len(agent_ids) < 2:
        return [
            VerificationCheck(
                name="topology.fault_tolerant",
                passed=True,
                severity="info",
                message="Single-agent app; fault tolerance trivially satisfied.",
                details={"agent_count": len(agent_ids)},
            )
        ]

    topology = manifest.get("topology") or {}
    graph = NetworkGraph()
    for aid in agent_ids:
        graph.add_node(aid)
    for conn in topology.get("connections") or []:
        src = _connection_from(conn)
        if src not in agent_ids:
            continue
        for tgt in _connection_to(conn):
            if tgt in agent_ids:
                graph.add_edge(src, tgt)

    if graph.edge_count == 0:
        return [
            VerificationCheck(
                name="topology.fault_tolerant",
                passed=True,
                severity="info",
                message="No A2A edges; fault tolerance does not apply.",
                details={"edge_count": 0},
            )
        ]

    connectivity = ConnectivityAnalyzer(graph).analyze()
    if not connectivity.articulation_points:
        return [
            VerificationCheck(
                name="topology.fault_tolerant",
                passed=True,
                severity="info",
                message="No articulation points; topology is fault-tolerant.",
                details={},
            )
        ]

    return [
        VerificationCheck(
            name="topology.fault_tolerant",
            passed=False,
            severity="warning",
            message=(
                f"Topology has {len(connectivity.articulation_points)} "
                "articulation point(s): removing any disconnects the graph. "
                "Acceptable for hub-and-spoke designs; flag for fault-"
                "tolerant ones."
            ),
            details={
                "articulation_points": list(connectivity.articulation_points),
                "bridges": [list(b) for b in connectivity.bridges],
            },
        )
    ]


# ----------------------------------------------------------------------
# Check: topology privilege flow (error severity)
# ----------------------------------------------------------------------


def _check_topology_privilege_flow(
    manifest: dict[str, Any],
) -> list[VerificationCheck]:
    """Verify cross-trust-boundary connections declare privilege attenuation.

    A connection that crosses from one trust boundary to another must
    have a non-empty ``privilege_attenuation`` mapping. The attenuation
    declares which capabilities are revoked when the privileged side
    invokes the less-privileged side (or vice versa), preventing
    capability escalation by association.
    """
    topology = manifest.get("topology") or {}
    boundaries = topology.get("trust") or []

    if not boundaries:
        return [
            VerificationCheck(
                name="topology.privilege_flow",
                passed=True,
                severity="info",
                message="No trust boundaries declared; privilege flow vacuous.",
                details={"boundary_count": 0},
            )
        ]

    boundary_of: dict[str, int] = {}
    for b_idx, boundary in enumerate(boundaries):
        for member in boundary.get("agents") or []:
            boundary_of[member] = b_idx

    violations: list[str] = []
    cross_boundary_edge_count = 0
    for idx, conn in enumerate(topology.get("connections") or []):
        src = _connection_from(conn)
        if not src or src not in boundary_of:
            continue
        src_boundary = boundary_of[src]
        for tgt in _connection_to(conn):
            if tgt not in boundary_of:
                continue
            if boundary_of[tgt] == src_boundary:
                continue
            cross_boundary_edge_count += 1
            if not (conn.get("privilege_attenuation") or {}):
                violations.append(
                    f"topology.connections[{idx}] crosses trust boundaries "
                    f"({src} -> {tgt}) without declaring "
                    "``privilege_attenuation``."
                )

    if not violations:
        return [
            VerificationCheck(
                name="topology.privilege_flow",
                passed=True,
                severity="info",
                message=(
                    f"All {cross_boundary_edge_count} cross-boundary "
                    "connection(s) declare privilege attenuation."
                ),
                details={
                    "boundary_count": len(boundaries),
                    "cross_boundary_edges": cross_boundary_edge_count,
                },
            )
        ]

    return [
        VerificationCheck(
            name="topology.privilege_flow",
            passed=False,
            severity="error",
            message=f"{len(violations)} cross-boundary edge(s) lack privilege attenuation",
            details={"violations": violations},
        )
    ]


# ----------------------------------------------------------------------
# Check: workflow deadlock freedom (error severity)
# ----------------------------------------------------------------------


def _check_workflow_deadlock_freedom(
    manifest: dict[str, Any],
) -> list[VerificationCheck]:
    """For each workflow, build a CSP process and assert deadlock freedom.

    Workflow-pattern dispatch:

    * ``fan_out_then_synthesize`` -> all non-final steps execute in
      parallel synchronized on completion (TICK), then the final step
      runs as a synthesis stage.
    * Any step with non-empty ``depends_on`` -> ``WorkflowSpec.from_dag``
      builds the precedence DAG from explicit dependencies.
    * Otherwise -> ``WorkflowSpec.sequential_tasks`` from step ordering.

    A step whose ``type == "parallel"`` (with ``agents: list[...]``)
    is expanded inside its position: each parallel target becomes an
    event composed with the others under TICK synchronization. The
    composite step then participates in the surrounding pipeline /
    DAG as a single event point.
    """
    workflows = manifest.get("workflows") or []
    if not workflows:
        return [
            VerificationCheck(
                name="workflow.deadlock_free",
                passed=True,
                severity="info",
                message="No workflows declared.",
                details={"workflow_count": 0},
            )
        ]

    checks: list[VerificationCheck] = []
    for wf in workflows:
        wf_id = wf.get("id", "<unnamed>")
        steps = wf.get("steps") or []
        if not steps:
            checks.append(
                VerificationCheck(
                    name="workflow.deadlock_free",
                    passed=True,
                    severity="info",
                    message=f"Workflow '{wf_id}' has no steps; trivially safe.",
                    details={"workflow_id": wf_id, "step_count": 0},
                )
            )
            continue

        try:
            process = _workflow_to_process(wf)
        except (KeyError, ValueError, TypeError) as exc:
            checks.append(
                VerificationCheck(
                    name="workflow.deadlock_free",
                    passed=False,
                    severity="error",
                    message=(
                        f"Workflow '{wf_id}': cannot build CSP process "
                        f"({type(exc).__name__}: {exc})"
                    ),
                    details={
                        "workflow_id": wf_id,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
            )
            continue

        lts = build_lts(process)
        deadlock_analysis = detect_deadlock(lts)
        common_details: dict[str, Any] = {
            "workflow_id": wf_id,
            "pattern": wf.get("pattern", "pipeline"),
            "step_count": len(steps),
            "lts_states": lts.num_states,
            "lts_transitions": lts.num_transitions,
        }

        if not deadlock_analysis.has_deadlock:
            checks.append(
                VerificationCheck(
                    name="workflow.deadlock_free",
                    passed=True,
                    severity="error",
                    message=(
                        f"Workflow '{wf_id}' is deadlock-free "
                        f"({lts.num_states} states, "
                        f"{lts.num_transitions} transitions)."
                    ),
                    details=common_details,
                )
            )
        else:
            failing_details = dict(common_details)
            failing_details["deadlocked_state_count"] = len(lts.deadlock_states())
            if deadlock_analysis.deadlock_traces:
                first_trace = deadlock_analysis.deadlock_traces[0]
                failing_details["deadlock_trace"] = [str(e) for e in first_trace]
            checks.append(
                VerificationCheck(
                    name="workflow.deadlock_free",
                    passed=False,
                    severity="error",
                    message=(
                        f"Workflow '{wf_id}' can deadlock "
                        f"({failing_details['deadlocked_state_count']} "
                        "deadlocked state(s) reachable)."
                    ),
                    details=failing_details,
                )
            )

    return checks


def _workflow_to_process(wf: dict[str, Any]) -> Process:
    """Convert a workflow dict into a CSP process.

    The returned process is the CSP behavioral model of the workflow's
    coordination shape; it does not model agent-internal execution.
    That is sufficient to detect coordination-level deadlocks (the
    failure mode this check targets). Per-step ``parallel`` types are
    expanded inline so each step contributes one CSP event point even
    when the underlying execution fans out across agents.
    """
    steps = wf["steps"]
    if not all(isinstance(s, dict) and s.get("id") for s in steps):
        raise ValueError("every workflow step must be a dict with an 'id' field")

    pattern = wf.get("pattern", "pipeline")

    # Expand parallel-type steps into composite CSP processes keyed by
    # the step's own ID. Each step is reduced to a single event for
    # the outer pipeline / DAG composition.
    step_processes: dict[str, Process] = {}
    for step in steps:
        step_processes[step["id"]] = _step_to_process(step)

    step_ids = [s["id"] for s in steps]

    if pattern == "fan_out_then_synthesize" and len(step_ids) > 1:
        body_ids = step_ids[:-1]
        synth_id = step_ids[-1]
        sync_set = frozenset({TICK})
        fan_out: Process = step_processes[body_ids[0]]
        for sid in body_ids[1:]:
            fan_out = parallel(fan_out, step_processes[sid], sync=sync_set)
        return sequential(fan_out, step_processes[synth_id])

    has_explicit_deps = any((s.get("depends_on") or []) for s in steps)
    if has_explicit_deps:
        nodes: dict[str, WorkflowNodeType] = {sid: WorkflowNodeType.TASK for sid in step_ids}
        edges: list[tuple[str, str]] = []
        for step in steps:
            for dep in step.get("depends_on") or []:
                edges.append((dep, step["id"]))
        spec = WorkflowSpec.from_dag(wf.get("id", "wf"), nodes, edges)
        return spec.process

    # Sequential pipeline: chain step CSP processes by Sequential
    # composition so per-step parallel sub-events stay inside their
    # own step boundary. SKIP terminates the chain so the final
    # Sequential composes correctly.
    sequenced: Process = skip()
    for sid in reversed(step_ids):
        sequenced = sequential(step_processes[sid], sequenced)
    return sequenced


def _step_to_process(step: dict[str, Any]) -> Process:
    """Build a CSP process for a single workflow step.

    A sequential step (the default) becomes ``Prefix(step_id, SKIP)`` --
    one event followed by termination. A ``type == "parallel"`` step
    with ``agents: list[{agent, action, ...}]`` becomes the parallel
    composition of one event per agent, synchronized on completion
    (TICK), then SKIP. The event names are ``{step_id}.{agent_id}``
    so the surrounding LTS analysis can distinguish sub-events from
    different parallel steps that share an agent.
    """
    sid = step["id"]
    parallel_targets = step.get("agents") or []

    if step.get("type") == "parallel" and parallel_targets:
        sync_set = frozenset({TICK})
        events = [
            prefix(f"{sid}.{_parallel_target_id(entry)}", skip()) for entry in parallel_targets
        ]
        composite: Process = events[0]
        for event_process in events[1:]:
            composite = parallel(composite, event_process, sync=sync_set)
        return composite

    return prefix(sid, skip())


def _parallel_target_id(entry: Any) -> str:
    """Extract an identifier for one parallel target of a ``parallel`` step.

    The target may be a dict with ``agent`` (the canonical key) or
    ``id``. Anything else is converted via ``str`` so a malformed entry
    surfaces as a uniquely-named event rather than a runtime error;
    reference integrity catches the malformation separately.
    """
    if isinstance(entry, dict):
        return str(entry.get("agent") or entry.get("id") or entry)
    return str(entry)


__all__ = [
    "Severity",
    "VerificationCheck",
    "VerificationReport",
    "verify",
]
