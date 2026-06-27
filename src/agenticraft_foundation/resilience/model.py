"""Swappable fault-tolerance models for the resilience diagnostic.

A :class:`ResilienceModel` maps an agent topology (a
:class:`~agenticraft_foundation.topology.NetworkGraph`) to the maximum number
of crash-stop and Byzantine faults the topology can structurally withstand.
The rule is isolated behind the protocol so the validated empirical model can
be plugged into :func:`~agenticraft_foundation.resilience.analyze_topology`
without touching the rest of the diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from agenticraft_foundation.topology import (
    ConnectivityAnalysis,
    ConnectivityAnalyzer,
    NetworkGraph,
    classical_fault_tolerance,
)

#: Provenance for the provisional placeholder model. Bump when the bounds change.
CLASSICAL_MODEL_VERSION = "classical-quorum-connectivity/0.1.0"


@dataclass(frozen=True)
class FaultEstimate:
    """A model's structural fault-tolerance estimate for one topology.

    Attributes:
        f_crash: Max crash-stop faults tolerated (agents that fail silently)
            while the system still completes — connectivity and quorum preserved.
        f_byz: Max Byzantine faults *structurally* tolerable. This is **capacity,
            not a guarantee**: it says the topology meets the necessary conditions
            (n>=3f+1, vertex connectivity κ>=2f+1), not that the system survives —
            realized Byzantine tolerance also needs a running voting/agreement
            mechanism over the redundant agents, which topology alone never
            establishes.
        crash_binding: Human-readable binding constraint for the crash model.
        byz_binding: Human-readable binding constraint for the Byzantine model.
        n_agents: Number of agents (graph nodes).
        vertex_connectivity: κ — minimum nodes to remove to disconnect the graph.
        algebraic_connectivity: λ₂ — reported only; not an input to the classical
            formulas (a likely input to the future empirical model).
        redundancy_score: Fraction of the fully-meshed edge budget present
            (``edge_count / (n*(n-1)/2)``), in ``[0, 1]`` — a "how meshed" signal.
        articulation_points: Agents whose removal disconnects the mesh.
        suggested_edges: Redundancy edges that would raise connectivity.
        cascade_grade: Optional empirically-grounded crash-stop cascade-risk
            band, set by the RD-28 EmpiricalMeshModel (None for the classical
            model). An annotation only — it never changes the integer bounds.
    """

    f_crash: int
    f_byz: int
    crash_binding: str
    byz_binding: str
    n_agents: int
    vertex_connectivity: int
    algebraic_connectivity: float
    redundancy_score: float
    articulation_points: list[str] = field(default_factory=list)
    suggested_edges: list[tuple[str, str]] = field(default_factory=list)
    cascade_grade: str | None = None


@runtime_checkable
class ResilienceModel(Protocol):
    """Maps a topology to a :class:`FaultEstimate`, with provenance.

    Implement this protocol and pass an instance to ``analyze_topology(graph,
    model=...)`` to swap the fault rule (e.g. the empirical RD-28 model) in
    without changing any other code.
    """

    model_version: str
    provisional: bool

    def estimate(self, graph: NetworkGraph) -> FaultEstimate:
        """Return the structural fault-tolerance estimate for ``graph``."""
        ...


class ClassicalQuorumModel:
    """Provisional placeholder using classical distributed-systems bounds.

    .. warning::

        This is the **classical distributed-systems approximation, NOT the
        validated AgentiCraft RD-28 empirical model** (the chained-GSM8K
        self-healing-mesh result; see ``docs/research/experiment_history.md``
        Phase 5). The numbers are textbook agreement bounds, not measured
        tolerances. Swap in the empirical model via ``analyze_topology(graph,
        model=...)`` once it lands. ``provisional`` is ``True`` and
        ``model_version`` records the provenance so output is never mistaken
        for the validated result.

    Bounds, computed over the **undirected projection** of the graph
    (``NetworkGraph`` bidirectionalizes edges; vertex connectivity κ and λ₂ are
    undirected notions), with ``n`` agents and κ = vertex connectivity::

        f_crash = max(0, min(κ - 1, (n - 1) // 2))  # stay connected AND n>=2f+1
        f_byz = max(0, min((κ - 1) // 2, (n - 1) // 3))  # κ>=2f+1 AND n>=3f+1

    NARRATIVE BOUNDARY (do not overclaim). Both bounds are monotone in κ and n,
    so this model **cannot** exhibit a topology that wins on Byzantine while
    losing on crash-stop — that crossover (the AgentiCraft iff result) needs the
    empirical model. The placeholder demonstrates *Byzantine-is-harder*
    (``f_byz <= f_crash`` always; the orchestrator-survives-0 demo), **not
    topology-crossover**. Do not frame demos as the iff result.

    CAPACITY, NOT GUARANTEE. ``f_byz`` is *structural capacity*: it means the
    topology satisfies the necessary conditions for surviving that many Byzantine
    faults, not that it does. Realized tolerance requires a running voting /
    agreement mechanism over the redundant agents. And because κ is undirected,
    ``f_byz`` cannot see directed source-multiplicity (whether a consumer has
    independent sources to vote across); it is a coarse structural proxy, not a
    dataflow-level BFT check.

    The bound formula itself lives in
    :func:`~agenticraft_foundation.topology.classical_fault_tolerance` — the
    single source of truth shared with
    :meth:`ConnectivityAnalyzer.analyze_fault_tolerance`, so the math primitive
    and this diagnostic always agree. This model adds the product layer on top:
    the binding-constraint narrative, the λ₂ metric, and the capacity caveats.
    """

    model_version: str = CLASSICAL_MODEL_VERSION
    provisional: bool = True

    def estimate(self, graph: NetworkGraph) -> FaultEstimate:
        n = graph.node_count
        spectral = graph.analyze()  # one eigendecomposition: λ₂ + suggested edges
        lambda2 = spectral.algebraic_connectivity
        max_edges = n * (n - 1) // 2
        redundancy_score = graph.edge_count / max_edges if max_edges > 0 else 0.0

        if n <= 1:
            reason = (
                "no agents in the topology" if n == 0 else "single agent — no redundancy possible"
            )
            return FaultEstimate(
                f_crash=0,
                f_byz=0,
                crash_binding=reason,
                byz_binding=reason,
                n_agents=n,
                vertex_connectivity=0,
                algebraic_connectivity=lambda2,
                redundancy_score=redundancy_score,
            )

        conn = ConnectivityAnalyzer(graph).analyze()
        kappa = conn.vertex_connectivity
        suggested = list(spectral.suggested_edges)

        if not conn.is_connected:
            return FaultEstimate(
                f_crash=0,
                f_byz=0,
                crash_binding="graph is disconnected — some agents are unreachable",
                byz_binding="graph is disconnected — no agreement is possible",
                n_agents=n,
                vertex_connectivity=kappa,
                algebraic_connectivity=lambda2,
                redundancy_score=redundancy_score,
                articulation_points=list(conn.articulation_points),
                suggested_edges=suggested,
            )

        # f values come from the shared source of truth. The per-model caps
        # below mirror its min() arguments and exist only to label which
        # constraint binds; a consistency test guards that they stay in step.
        f_crash, f_byz = classical_fault_tolerance(n, kappa)
        crash_caps = (kappa - 1, (n - 1) // 2)
        byz_caps = ((kappa - 1) // 2, (n - 1) // 3)

        return FaultEstimate(
            f_crash=f_crash,
            f_byz=f_byz,
            crash_binding=_binding(*crash_caps, n, conn, kind="crash"),
            byz_binding=_binding(*byz_caps, n, conn, kind="byzantine"),
            n_agents=n,
            vertex_connectivity=kappa,
            algebraic_connectivity=lambda2,
            redundancy_score=redundancy_score,
            articulation_points=list(conn.articulation_points),
            suggested_edges=suggested,
        )


def _binding(
    conn_term: int,
    quorum_term: int,
    n: int,
    conn: ConnectivityAnalysis,
    *,
    kind: str,
) -> str:
    """Name the binding constraint: connectivity (structure) or quorum (count).

    Connectivity binds when the κ-derived term is the smaller one; on a tie we
    attribute it to connectivity only if an articulation point exists (a
    well-connected graph at the same value is really replica-count limited).
    """
    connectivity_bound = conn_term < quorum_term or (
        conn_term == quorum_term and bool(conn.articulation_points)
    )
    if connectivity_bound:
        aps = conn.articulation_points
        if aps:
            if len(aps) == 1:
                return (
                    f"single hub / articulation point: {aps[0]} (removing it disconnects the mesh)"
                )
            return (
                f"{len(aps)} articulation points ({', '.join(aps[:5])}) "
                "— removing any disconnects the mesh"
            )
        need = (
            "2f+1 independent agreement paths"
            if kind == "byzantine"
            else "a higher vertex connectivity"
        )
        return f"limited redundant connectivity (κ={conn.vertex_connectivity}); needs {need}"
    if kind == "byzantine":
        return f"no voting quorum — only {n} agents; Byzantine agreement needs >= 3f+1"
    return f"replica quorum — only {n} agents; crash-stop agreement needs >= 2f+1"


__all__ = [
    "CLASSICAL_MODEL_VERSION",
    "ClassicalQuorumModel",
    "FaultEstimate",
    "ResilienceModel",
]
