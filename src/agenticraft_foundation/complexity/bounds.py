"""Complexity bounds and theoretical limits.

This module provides utilities for reasoning about complexity bounds,
including lower bounds, optimality proofs, and theoretical limits
for distributed algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BoundType(str, Enum):
    """Type of complexity bound."""

    UPPER = "upper"  # O notation
    LOWER = "lower"  # Omega notation
    TIGHT = "tight"  # Theta notation
    EXPECTED = "expected"  # Average case
    AMORTIZED = "amortized"  # Amortized analysis


@dataclass(frozen=True)
class TheoreticalBound:
    """A theoretical bound from distributed systems literature.

    Attributes:
        problem: The problem this bound applies to
        metric: What is being bounded (time, messages, rounds)
        bound_type: Type of bound (upper, lower, tight)
        expression: The bound expression
        conditions: Conditions under which bound holds
        source: Citation for the bound
        proof_sketch: Brief description of proof technique
    """

    problem: str
    metric: str
    bound_type: BoundType
    expression: str
    conditions: list[str] = field(default_factory=list)
    source: str = ""
    proof_sketch: str = ""

    def __str__(self) -> str:
        bound_symbols = {
            BoundType.UPPER: "O",
            BoundType.LOWER: "Ω",
            BoundType.TIGHT: "Θ",
            BoundType.EXPECTED: "E",
            BoundType.AMORTIZED: "O*",
        }
        symbol = bound_symbols.get(self.bound_type, "?")
        return f"{self.problem} {self.metric}: {symbol}({self.expression})"


# Well-known theoretical bounds from distributed systems literature
CONSENSUS_BOUNDS = {
    "synchronous_consensus": TheoreticalBound(
        problem="Synchronous Consensus",
        metric="rounds",
        bound_type=BoundType.TIGHT,
        expression="f + 1",
        conditions=["f crash failures", "n >= f + 1"],
        source="Fischer & Lynch 1985",
        proof_sketch="f+1 rounds necessary and sufficient to tolerate f crashes",
    ),
    "async_consensus_impossibility": TheoreticalBound(
        problem="Asynchronous Consensus",
        metric="termination",
        bound_type=BoundType.LOWER,
        expression="impossible",
        conditions=["asynchronous", "even 1 crash failure"],
        source="Fischer, Lynch, Paterson 1985 (FLP)",
        proof_sketch="Bivalent configurations can be extended indefinitely",
    ),
    "byzantine_consensus_nodes": TheoreticalBound(
        problem="Byzantine Consensus",
        metric="nodes",
        bound_type=BoundType.LOWER,
        expression="3f + 1",
        conditions=["f Byzantine failures"],
        source="Lamport, Shostak, Pease 1982",
        proof_sketch="Need 2f+1 honest nodes to reach agreement despite f liars",
    ),
    "byzantine_consensus_messages": TheoreticalBound(
        problem="Byzantine Consensus",
        metric="messages",
        bound_type=BoundType.LOWER,
        expression="n^2",
        conditions=["f < n/3 Byzantine", "authenticated channels"],
        source="Dolev & Reischuk 1985",
        proof_sketch="All-to-all communication required for Byzantine agreement",
    ),
    "authenticated_byzantine": TheoreticalBound(
        problem="Authenticated Byzantine Consensus",
        metric="messages",
        bound_type=BoundType.TIGHT,
        expression="n",
        conditions=["digital signatures", "f < n/3 Byzantine"],
        source="HotStuff, Yin et al. 2019",
        proof_sketch="Linear messages with threshold signatures",
    ),
}

GOSSIP_BOUNDS = {
    "epidemic_convergence": TheoreticalBound(
        problem="Epidemic Dissemination",
        metric="rounds",
        bound_type=BoundType.EXPECTED,
        expression="log n",
        conditions=["push gossip", "fanout >= 2"],
        source="Demers et al. 1987",
        proof_sketch="Exponential growth until n/2, then exponential decay",
    ),
    "epidemic_messages": TheoreticalBound(
        problem="Epidemic Dissemination",
        metric="messages",
        bound_type=BoundType.EXPECTED,
        expression="n log n",
        conditions=["push-pull gossip"],
        source="Karp et al. 2000",
        proof_sketch="Each node sends O(log n) messages",
    ),
}

LEADER_ELECTION_BOUNDS = {
    "ring_comparison": TheoreticalBound(
        problem="Ring Leader Election",
        metric="messages",
        bound_type=BoundType.TIGHT,
        expression="n log n",
        conditions=["comparison-based", "unidirectional ring"],
        source="Frederickson & Lynch 1987",
        proof_sketch="Information-theoretic lower bound from sorting",
    ),
    "complete_graph": TheoreticalBound(
        problem="Complete Graph Leader Election",
        metric="messages",
        bound_type=BoundType.TIGHT,
        expression="n",
        conditions=["unique IDs", "complete graph"],
        source="Korach et al. 1984",
    ),
}

BROADCAST_BOUNDS = {
    "reliable_broadcast": TheoreticalBound(
        problem="Reliable Broadcast",
        metric="messages",
        bound_type=BoundType.LOWER,
        expression="n^2",
        conditions=["f < n/3 Byzantine"],
        source="Bracha 1987",
        proof_sketch="Echo-echo pattern requires all-to-all",
    ),
    "best_effort_broadcast": TheoreticalBound(
        problem="Best-Effort Broadcast",
        metric="messages",
        bound_type=BoundType.TIGHT,
        expression="n",
        conditions=["no failures"],
        source="Basic result",
    ),
}


MESH_COMMUNICATION_BOUNDS = {
    "full_mesh_messages": TheoreticalBound(
        problem="Full Mesh Communication",
        metric="messages",
        bound_type=BoundType.TIGHT,
        expression="n",
        conditions=["fully connected topology", "single broadcast"],
        source="Standard result",
        proof_sketch="Each node sends one message; all-to-all in parallel",
    ),
    "full_mesh_rounds": TheoreticalBound(
        problem="Full Mesh Communication",
        metric="rounds",
        bound_type=BoundType.TIGHT,
        expression="1",
        conditions=["fully connected topology"],
        source="Standard result",
    ),
    "k_regular_messages": TheoreticalBound(
        problem="k-Regular Mesh Communication",
        metric="messages",
        bound_type=BoundType.UPPER,
        expression="n * k",
        conditions=["k-regular graph", "flooding"],
        source="Graph theory standard",
        proof_sketch="Each node forwards to k neighbors",
    ),
    "k_regular_rounds": TheoreticalBound(
        problem="k-Regular Mesh Communication",
        metric="rounds",
        bound_type=BoundType.UPPER,
        expression="D",
        conditions=["k-regular graph", "D = diameter"],
        source="Graph theory standard",
        proof_sketch="Information propagates one hop per round",
    ),
    "tree_messages": TheoreticalBound(
        problem="Tree Topology Communication",
        metric="messages",
        bound_type=BoundType.TIGHT,
        expression="n",
        conditions=["tree topology", "convergecast or broadcast"],
        source="Standard result",
        proof_sketch="Each edge carries exactly one message",
    ),
    "tree_rounds": TheoreticalBound(
        problem="Tree Topology Communication",
        metric="rounds",
        bound_type=BoundType.TIGHT,
        expression="log n",
        conditions=["balanced tree", "depth = O(log n)"],
        source="Standard result",
    ),
    "ring_messages": TheoreticalBound(
        problem="Ring Topology Communication",
        metric="messages",
        bound_type=BoundType.TIGHT,
        expression="n",
        conditions=["unidirectional ring"],
        source="Standard result",
    ),
    "ring_rounds": TheoreticalBound(
        problem="Ring Topology Communication",
        metric="rounds",
        bound_type=BoundType.TIGHT,
        expression="n",
        conditions=["unidirectional ring"],
        source="Standard result",
        proof_sketch="Message must traverse entire ring",
    ),
    "star_messages": TheoreticalBound(
        problem="Star (Hub-and-Spoke) Communication",
        metric="messages",
        bound_type=BoundType.TIGHT,
        expression="n",
        conditions=["star topology", "hub broadcasts to all spokes"],
        source="Standard result",
    ),
    "star_rounds": TheoreticalBound(
        problem="Star (Hub-and-Spoke) Communication",
        metric="rounds",
        bound_type=BoundType.TIGHT,
        expression="2",
        conditions=["star topology", "gather + broadcast"],
        source="Standard result",
        proof_sketch="One round to hub, one round from hub",
    ),
    "byzantine_consensus_messages": TheoreticalBound(
        problem="Byzantine Consensus (Mesh)",
        metric="messages",
        bound_type=BoundType.TIGHT,
        expression="n^2",
        conditions=["f < n/3 Byzantine", "full mesh"],
        source="Dolev & Reischuk 1985",
        proof_sketch="All-to-all required for Byzantine agreement",
    ),
}


@dataclass
class OptimalityCheck:
    """Result of checking if an algorithm is optimal.

    Attributes:
        is_optimal: Whether algorithm matches lower bound
        algorithm_bound: The algorithm's complexity
        lower_bound: The theoretical lower bound
        gap: Gap between algorithm and lower bound
        conditions_match: Whether conditions match
        notes: Additional notes
    """

    is_optimal: bool
    algorithm_bound: str
    lower_bound: str
    gap: str | None = None
    conditions_match: bool = True
    notes: str = ""


def check_optimality(
    algorithm_complexity: str,
    problem: str,
    metric: str = "messages",
) -> OptimalityCheck:
    """Check if an algorithm achieves optimal complexity.

    Args:
        algorithm_complexity: The algorithm's complexity (e.g., "O(n^2)")
        problem: The problem being solved
        metric: The metric to check

    Returns:
        OptimalityCheck with results
    """
    # Collect all bounds
    all_bounds = {
        **CONSENSUS_BOUNDS,
        **GOSSIP_BOUNDS,
        **LEADER_ELECTION_BOUNDS,
        **BROADCAST_BOUNDS,
        **MESH_COMMUNICATION_BOUNDS,
    }

    # Find matching lower bound
    matching_bound = None
    for _key, bound in all_bounds.items():
        if (
            problem.lower() in bound.problem.lower()
            and bound.metric.lower() == metric.lower()
            and bound.bound_type in (BoundType.LOWER, BoundType.TIGHT)
        ):
            matching_bound = bound
            break

    if not matching_bound:
        return OptimalityCheck(
            is_optimal=False,
            algorithm_bound=algorithm_complexity,
            lower_bound="unknown",
            notes=f"No known lower bound for {problem} {metric}",
        )

    # Simple comparison (would need more sophisticated parsing for full analysis)
    algo_upper = algorithm_complexity.lower().replace("o(", "").replace(")", "")
    lower_expr = matching_bound.expression.lower()

    is_optimal = algo_upper == lower_expr or (
        algo_upper in ["n", "n log n", "n^2"] and algo_upper == lower_expr
    )

    gap = (
        None
        if is_optimal
        else f"Algorithm: {algorithm_complexity}, Lower: Ω({matching_bound.expression})"
    )

    return OptimalityCheck(
        is_optimal=is_optimal,
        algorithm_bound=algorithm_complexity,
        lower_bound=f"Ω({matching_bound.expression})",
        gap=gap,
        notes=f"Source: {matching_bound.source}",
    )


@dataclass
class ImpossibilityResult:
    """Represents an impossibility result.

    Attributes:
        problem: The impossible problem
        conditions: Conditions making it impossible
        source: Citation
        workarounds: Known workarounds that weaken assumptions
    """

    problem: str
    conditions: list[str]
    source: str
    workarounds: list[str] = field(default_factory=list)


# Famous impossibility results
IMPOSSIBILITY_RESULTS = [
    ImpossibilityResult(
        problem="Deterministic Consensus in Asynchronous Systems",
        conditions=[
            "Asynchronous timing model",
            "At least one crash failure possible",
            "Deterministic algorithm",
        ],
        source="Fischer, Lynch, Paterson 1985 (FLP Impossibility)",
        workarounds=[
            "Use randomization (Ben-Or protocol)",
            "Assume partial synchrony (DLS, PBFT)",
            "Use failure detectors (Chandra-Toueg)",
            "Allow probabilistic termination",
        ],
    ),
    ImpossibilityResult(
        problem="Byzantine Agreement with n <= 3f",
        conditions=[
            "n <= 3f nodes",
            "f Byzantine failures",
            "No authentication/signatures",
        ],
        source="Lamport, Shostak, Pease 1982",
        workarounds=[
            "Add digital signatures (allows n >= 2f+1)",
            "Use trusted hardware",
            "Reduce Byzantine to crash failures",
        ],
    ),
    ImpossibilityResult(
        problem="Wait-Free Consensus with Compare-And-Swap",
        conditions=[
            "Wait-free termination required",
            "Only compare-and-swap registers",
            "n > 2 processes",
        ],
        source="Herlihy 1991 (Consensus Number)",
        workarounds=[
            "Use universal constructions",
            "Relax to lock-free",
            "Use stronger primitives (e.g., LL/SC)",
        ],
    ),
    ImpossibilityResult(
        problem="Exactly-Once Delivery in Unreliable Networks",
        conditions=[
            "Network can lose messages",
            "No upper bound on delays",
            "Processes can crash",
        ],
        source="Two Generals Problem (common knowledge)",
        workarounds=[
            "At-least-once with idempotency",
            "At-most-once with acknowledgments",
            "Use reliable broadcast primitives",
        ],
    ),
]


def get_impossibility(problem: str) -> ImpossibilityResult | None:
    """Get impossibility result for a problem.

    Args:
        problem: Problem description

    Returns:
        ImpossibilityResult if found, None otherwise
    """
    for result in IMPOSSIBILITY_RESULTS:
        if problem.lower() in result.problem.lower():
            return result
    return None


def validate_fault_tolerance(
    n: int,
    f: int,
    fault_model: str,
) -> tuple[bool, str]:
    """Validate if fault tolerance is achievable.

    Args:
        n: Number of nodes
        f: Number of failures to tolerate
        fault_model: Type of failures (crash, byzantine)

    Returns:
        Tuple of (is_valid, explanation)
    """
    if fault_model.lower() in ("byzantine", "bft"):
        required = 3 * f + 1
        if n < required:
            return (
                False,
                f"Byzantine agreement requires n >= 3f+1."
                f" With f={f}, need n>={required}, have n={n}",
            )
        return (
            True,
            f"Valid: {n} nodes can tolerate {f} Byzantine failures (need {required})",
        )

    elif fault_model.lower() in ("crash", "crash_stop", "cfr"):
        required = 2 * f + 1
        if n < required:
            return (
                False,
                f"Crash tolerance requires n >= 2f+1. With f={f}, need n>={required}, have n={n}",
            )
        return (
            True,
            f"Valid: {n} nodes can tolerate {f} crash failures (need {required})",
        )

    return (False, f"Unknown fault model: {fault_model}")


@dataclass
class ComplexityComparison:
    """Comparison of multiple algorithm complexities.

    Attributes:
        problem: The problem being solved
        algorithms: Dict mapping algorithm name to complexity
        optimal_for_metric: Dict mapping metric to best algorithm
        pareto_optimal: Algorithms that are Pareto optimal
    """

    problem: str
    algorithms: dict[str, dict[str, str]]
    optimal_for_metric: dict[str, list[str]] = field(default_factory=dict)
    pareto_optimal: list[str] = field(default_factory=list)


def compare_algorithms(
    problem: str,
    algorithms: dict[str, dict[str, str]],
) -> ComplexityComparison:
    """Compare complexity of multiple algorithms.

    Args:
        problem: The problem being solved
        algorithms: Dict mapping algorithm name to complexity dict
            Example: {"PBFT": {"messages": "O(n^2)", "rounds": "O(1)"}}

    Returns:
        ComplexityComparison with analysis
    """
    # Find optimal for each metric
    metrics: set[str] = set()
    for algo_metrics in algorithms.values():
        metrics.update(algo_metrics.keys())

    optimal_for: dict[str, list[str]] = {}

    # Simple ordering (would need proper parsing for full comparison)
    complexity_order = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)"]

    def get_order(c: str) -> int:
        try:
            return complexity_order.index(c)
        except ValueError:
            return 100  # Unknown complexity

    for metric in metrics:
        # Get all algorithms with this metric
        algo_values = [
            (name, algo_metrics.get(metric, ""))
            for name, algo_metrics in algorithms.items()
            if metric in algo_metrics
        ]

        if not algo_values:
            continue

        sorted_algos = sorted(algo_values, key=lambda x: get_order(x[1]))
        if sorted_algos:
            best_complexity = sorted_algos[0][1]
            optimal_for[metric] = [name for name, c in sorted_algos if c == best_complexity]

    # Find Pareto optimal (no algorithm dominates in all metrics)
    pareto: list[str] = []
    for name in algorithms:
        dominated = False
        for other_name in algorithms:
            if name == other_name:
                continue
            # Check if other dominates name
            all_better_or_equal = True
            any_strictly_better = False
            for metric in metrics:
                name_c = algorithms[name].get(metric)
                other_c = algorithms[other_name].get(metric)
                if name_c and other_c:
                    name_idx = complexity_order.index(name_c) if name_c in complexity_order else 100
                    other_idx = (
                        complexity_order.index(other_c) if other_c in complexity_order else 100
                    )
                    if other_idx > name_idx:
                        all_better_or_equal = False
                    elif other_idx < name_idx:
                        any_strictly_better = True
            if all_better_or_equal and any_strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(name)

    return ComplexityComparison(
        problem=problem,
        algorithms=algorithms,
        optimal_for_metric=optimal_for,
        pareto_optimal=pareto,
    )


__all__ = [
    "BoundType",
    "TheoreticalBound",
    "CONSENSUS_BOUNDS",
    "GOSSIP_BOUNDS",
    "LEADER_ELECTION_BOUNDS",
    "BROADCAST_BOUNDS",
    "MESH_COMMUNICATION_BOUNDS",
    "OptimalityCheck",
    "check_optimality",
    "ImpossibilityResult",
    "IMPOSSIBILITY_RESULTS",
    "get_impossibility",
    "validate_fault_tolerance",
    "ComplexityComparison",
    "compare_algorithms",
]
