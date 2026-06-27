"""The resilience diagnostic: analyze a topology, report fault tolerance.

Public surface::

    from agenticraft_foundation.resilience import analyze_topology, ResilienceTarget

    report = analyze_topology(graph, target=ResilienceTarget(byzantine=1, crash=2))
    report.f_crash, report.f_byz  # headline numbers (they usually differ)
    report.meets_target  # bool — gate CI on this
    report.model_version  # provenance of the rule used
    print(report.to_text())  # human-readable
    report.to_dict()  # structured / JSON
"""

from __future__ import annotations

from agenticraft_foundation.resilience.model import (
    ClassicalQuorumModel,
    FaultEstimate,
    ResilienceModel,
)
from agenticraft_foundation.resilience.report import ResilienceReport, ResilienceTarget
from agenticraft_foundation.topology import NetworkGraph


def analyze_topology(
    graph: NetworkGraph,
    *,
    target: ResilienceTarget | None = None,
    model: ResilienceModel | None = None,
) -> ResilienceReport:
    """Statically analyze an agent topology's fault tolerance.

    Args:
        graph: The agent topology — nodes are agents, edges are
            dependency / delegation / communication relationships. Build one
            directly (``NetworkGraph.create_star(...)``) or from a manifest via
            :func:`~agenticraft_foundation.resilience.graph_from_manifest`.
        target: Optional desired tolerance. When set, ``meets_target``,
            cost multipliers, and ordered remediations are populated.
        model: The fault rule. Defaults to the provisional
            :class:`~agenticraft_foundation.resilience.model.ClassicalQuorumModel`;
            pass the empirical model here once it lands.

    Returns:
        A :class:`ResilienceReport`. ``f_byz`` is structural capacity, not a
        realized-BFT guarantee.
    """
    model = model or ClassicalQuorumModel()
    estimate = model.estimate(graph)

    meets_target = True
    crash_cost: int | None = None
    byz_cost: int | None = None
    min_agents: int | None = None
    remediations: list[str] = []
    if target is not None:
        meets_target = estimate.f_crash >= target.crash and estimate.f_byz >= target.byzantine
        crash_cost = 2 * target.crash + 1 if target.crash > 0 else None
        byz_cost = 3 * target.byzantine + 1 if target.byzantine > 0 else None
        # Min replicas to satisfy BOTH quorums (crash n>=2c+1, Byzantine n>=3b+1).
        min_agents = max(2 * target.crash + 1, 3 * target.byzantine + 1)
        remediations = _remediations(estimate, target)

    return ResilienceReport(
        f_crash=estimate.f_crash,
        f_byz=estimate.f_byz,
        crash_binding=estimate.crash_binding,
        byz_binding=estimate.byz_binding,
        n_agents=estimate.n_agents,
        vertex_connectivity=estimate.vertex_connectivity,
        algebraic_connectivity=estimate.algebraic_connectivity,
        redundancy_score=estimate.redundancy_score,
        articulation_points=list(estimate.articulation_points),
        model_version=model.model_version,
        provisional=model.provisional,
        target=target,
        meets_target=meets_target,
        crash_cost_multiplier=crash_cost,
        byz_cost_multiplier=byz_cost,
        min_agents_for_target=min_agents,
        remediations=remediations,
    )


def _remediations(estimate: FaultEstimate, target: ResilienceTarget) -> list[str]:
    """Ordered, honestly-templated suggestions to reach ``target``.

    Crash-stop first (cheaper redundancy), then Byzantine (which additionally
    requires a realized voting quorum). Each suggestion is tied to the binding
    constraint; these are mechanical templates, not bespoke structural advice.
    """
    out: list[str] = []
    edges = _edges_hint(estimate.suggested_edges)

    if estimate.f_crash < target.crash:
        need_n, need_k = 2 * target.crash + 1, target.crash + 1
        if estimate.n_agents < need_n:
            out.append(
                f"Add {need_n - estimate.n_agents} more agent(s) to reach n >= {need_n} "
                f"(2*{target.crash}+1) for crash-stop tolerance {target.crash}."
            )
        if estimate.vertex_connectivity < need_k:
            if estimate.articulation_points:
                out.append(
                    "Remove the single-hub dependency at "
                    f"{estimate.articulation_points[0]} and add redundant paths so "
                    f"vertex connectivity κ >= {need_k}.{edges}"
                )
            else:
                out.append(
                    f"Add redundant connections to raise κ to >= {need_k} "
                    f"(2*{target.crash}+1 disjoint paths).{edges}"
                )

    if estimate.f_byz < target.byzantine:
        need_n, need_k = 3 * target.byzantine + 1, 2 * target.byzantine + 1
        out.append(
            f"Run a replicated + majority-vote ensemble of >= {need_n} agents "
            f"(3*{target.byzantine}+1) on the critical step — Byzantine tolerance "
            "requires a realized voting quorum, which topology alone does not provide."
        )
        if estimate.n_agents < need_n:
            out.append(f"Scale to n >= {need_n} agents (currently {estimate.n_agents}).")
        if estimate.vertex_connectivity < need_k:
            out.append(
                f"Raise κ to >= {need_k} (2*{target.byzantine}+1) via redundant "
                f"connections so voters have independent paths.{edges}"
            )

    return out


def _edges_hint(suggested: list[tuple[str, str]]) -> str:
    """Format up to three suggested edges as an inline hint."""
    if not suggested:
        return ""
    rendered = ", ".join(f"{src}--{tgt}" for src, tgt in suggested[:3])
    return f" Suggested edges: {rendered}."


__all__ = ["analyze_topology"]
