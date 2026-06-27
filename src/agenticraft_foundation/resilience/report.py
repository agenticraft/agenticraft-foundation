"""Output contract for the resilience diagnostic.

:class:`ResilienceTarget` is the desired tolerance to check against;
:class:`ResilienceReport` is the result of
:func:`~agenticraft_foundation.resilience.analyze_topology`. Both are frozen
dataclasses — the stable, JSON-renderable surface a developer or CI consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: One-line caveat rendered beside every Byzantine number. ``f_byz`` is a
#: structural-capacity figure, never a realized-BFT guarantee.
BYZ_CAVEAT = (
    "structural capacity — contingent on a realized voting/agreement mechanism; not a guarantee"
)


@dataclass(frozen=True)
class ResilienceTarget:
    """Fault tolerance the topology is checked against.

    Attributes:
        crash: Required crash-stop tolerance (agents that may fail silently).
        byzantine: Required Byzantine tolerance (agents that may return wrong
            or adversarial output).
    """

    crash: int = 0
    byzantine: int = 0


@dataclass(frozen=True)
class ResilienceReport:
    """Result of :func:`analyze_topology` — headline numbers plus diagnosis.

    Leads with ``f_crash`` and ``f_byz`` (they usually differ — that gap is the
    point). ``f_byz`` is *structural capacity*, not a realized-BFT guarantee;
    ``byz_semantics`` records this for machine consumers so the bare integer is
    never read as protection. See
    :class:`~agenticraft_foundation.resilience.model.ClassicalQuorumModel`.
    """

    f_crash: int
    f_byz: int
    crash_binding: str
    byz_binding: str
    n_agents: int
    vertex_connectivity: int
    algebraic_connectivity: float
    redundancy_score: float
    articulation_points: list[str]
    model_version: str
    provisional: bool
    target: ResilienceTarget | None = None
    meets_target: bool = True
    crash_cost_multiplier: int | None = None
    byz_cost_multiplier: int | None = None
    min_agents_for_target: int | None = None
    remediations: list[str] = field(default_factory=list)
    byz_semantics: str = "capacity"

    def to_dict(self) -> dict[str, Any]:
        """Render the report as a JSON-serializable dict."""
        return {
            "model_version": self.model_version,
            "provisional": self.provisional,
            "n_agents": self.n_agents,
            "vertex_connectivity": self.vertex_connectivity,
            "algebraic_connectivity": self.algebraic_connectivity,
            "redundancy_score": self.redundancy_score,
            "f_crash": self.f_crash,
            "f_byz": self.f_byz,
            "byz_semantics": self.byz_semantics,
            "byz_caveat": BYZ_CAVEAT,
            "byz_basis": "undirected-projection",
            "crash_binding": self.crash_binding,
            "byz_binding": self.byz_binding,
            "articulation_points": list(self.articulation_points),
            "target": (
                None
                if self.target is None
                else {"crash": self.target.crash, "byzantine": self.target.byzantine}
            ),
            "meets_target": self.meets_target,
            "min_agents_for_target": self.min_agents_for_target,
            "crash_cost_multiplier": self.crash_cost_multiplier,
            "byz_cost_multiplier": self.byz_cost_multiplier,
            "remediations": list(self.remediations),
        }

    def to_text(self) -> str:
        """Render a human-readable report, leading with the two numbers."""
        prov = " — PROVISIONAL placeholder" if self.provisional else ""
        lines = [
            f"Resilience diagnostic (model: {self.model_version}{prov})",
            "=" * 72,
            f"Agents: {self.n_agents}  |  vertex connectivity κ={self.vertex_connectivity}"
            f"  |  λ₂={self.algebraic_connectivity:.4f}"
            f"  |  redundancy={self.redundancy_score:.0%}",
            "",
            f"  Crash-stop tolerance   f_crash = {self.f_crash}",
            f"      binding: {self.crash_binding}",
            f"  Byzantine tolerance    f_byz   = {self.f_byz}",
            f"      [{BYZ_CAVEAT}]",
            f"      binding: {self.byz_binding}",
        ]

        if self.target is not None:
            status = "MET" if self.meets_target else "NOT MET"
            lines += [
                "",
                f"Target: crash>={self.target.crash}, "
                f"byzantine>={self.target.byzantine}  ->  {status}",
            ]
            if self.min_agents_for_target is not None:
                lines.append(f"  Minimum agents to meet target: {self.min_agents_for_target}")
            if self.meets_target and self.target.byzantine > 0:
                # The strongest positive signal the tool emits — never let it
                # read as "you are protected" without the capacity reminder.
                lines.append(
                    "  (Byzantine target MET = structural capacity only; confirm a "
                    "voting/agreement mechanism is actually running.)"
                )
            if self.byz_cost_multiplier:
                lines.append(
                    f"  Byzantine target implies ~{self.byz_cost_multiplier}x execution "
                    "cost (the voted region runs at that redundancy)."
                )
            if self.crash_cost_multiplier:
                lines.append(
                    f"  Crash target implies ~{self.crash_cost_multiplier}x execution cost."
                )
            if self.remediations:
                lines.append("  Remediations (in order):")
                lines += [f"    {i}. {r}" for i, r in enumerate(self.remediations, 1)]

        lines += [
            "",
            "Note: f_byz is structural capacity over the undirected communication graph,",
            "not a realized-BFT guarantee (which also needs a running voting mechanism).",
        ]
        if self.provisional:
            lines.append(
                "This is the classical placeholder, NOT the validated RD-28 empirical model."
            )
        return "\n".join(lines)

    def summary(self) -> str:
        """Alias for :meth:`to_text` (foundation report convention)."""
        return self.to_text()


__all__ = ["BYZ_CAVEAT", "ResilienceReport", "ResilienceTarget"]
