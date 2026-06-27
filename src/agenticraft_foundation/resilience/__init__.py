"""Static resilience diagnostic for multi-agent topologies.

Point it at an agent topology (a
:class:`~agenticraft_foundation.topology.NetworkGraph`, or an app manifest via
:func:`graph_from_manifest`) and it reports how many crash-stop faults the
topology can withstand and how many Byzantine faults its structure could support
(capacity, not a guarantee), the binding structural constraint, the redundancy +
execution-cost needed to hit a target, and ordered remediations.

Demo — an orchestrator / star survives **zero** Byzantine faults:

    >>> from agenticraft_foundation.topology import NetworkGraph
    >>> from agenticraft_foundation.resilience import (
    ...     analyze_topology,
    ...     ResilienceTarget,
    ... )
    >>> report = analyze_topology(
    ...     NetworkGraph.create_star(6), target=ResilienceTarget(byzantine=1)
    ... )
    >>> report.f_crash, report.f_byz
    (0, 0)
    >>> report.meets_target
    False
    >>> "single hub" in report.crash_binding
    True

NARRATIVE BOUNDARY — do not overclaim. The default
:class:`~agenticraft_foundation.resilience.model.ClassicalQuorumModel` is a
*provisional placeholder* using classical agreement bounds. Its ``f_crash`` and
``f_byz`` are both monotone in connectivity and node count, so it demonstrates
that **Byzantine is a separate, harder failure mode** (``f_byz <= f_crash``; the
orchestrator-survives-0 result) — it does **not**, and structurally cannot,
demonstrate that **a different topology wins for Byzantine than for crash-stop**
(the AgentiCraft iff result), which requires the empirical RD-28 model. Keep
demos to the single-point-of-failure / quorum claim.

CAPACITY, NOT GUARANTEE. ``f_byz`` is *structural capacity*: it says the topology
meets the necessary conditions to survive that many Byzantine faults, not that it
does. Realized Byzantine tolerance also needs a running voting/agreement
mechanism over the redundant agents, which topology alone never establishes.
"""

from __future__ import annotations

from agenticraft_foundation.resilience.analysis import analyze_topology
from agenticraft_foundation.resilience.loader import (
    graph_from_manifest,
    load_topology,
)
from agenticraft_foundation.resilience.model import (
    CLASSICAL_MODEL_VERSION,
    ClassicalQuorumModel,
    FaultEstimate,
    ResilienceModel,
)
from agenticraft_foundation.resilience.report import (
    ResilienceReport,
    ResilienceTarget,
)

__all__ = [
    # Entry point + report
    "analyze_topology",
    "ResilienceTarget",
    "ResilienceReport",
    # Swappable fault model
    "ResilienceModel",
    "ClassicalQuorumModel",
    "FaultEstimate",
    "CLASSICAL_MODEL_VERSION",
    # Manifest adapter
    "graph_from_manifest",
    "load_topology",
]
