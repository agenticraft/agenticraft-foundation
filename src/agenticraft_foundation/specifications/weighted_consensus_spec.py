"""Weighted Byzantine Fault Tolerance specifications.

Extends consensus specifications with quality-weighted quorum consensus.

Key concepts:
- Each agent has a weight w_i based on historical reliability
- Weighted quorum requires total weight >= 2W/3 (not just count)
- Weight-based leader rotation for liveness

References:
- Cachin et al. (2000) - Random oracles in Constantinople
- Malkhi & Reiter (1998) - Byzantine quorum systems
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agenticraft_foundation.specifications.consensus_spec import (
    ConsensusState,
    FormalProperty,
    PropertyResult,
    PropertyStatus,
    PropertyType,
)

logger = logging.getLogger(__name__)


@dataclass
class WeightedConsensusState:
    """Consensus state with per-agent quality weights.

    Extends ConsensusState with weight information for
    quality-weighted quorum consensus.
    """

    consensus_state: ConsensusState[Any]
    """Underlying consensus state"""

    weights: dict[str, float] = field(default_factory=dict)
    """Per-agent quality weights (w_i). Higher = more reliable."""

    weight_history: dict[str, list[float]] = field(default_factory=dict)
    """Historical weight records per agent"""

    @property
    def total_weight(self) -> float:
        """Total weight W = sum(w_i)."""
        return sum(self.weights.get(pid, 1.0) for pid in self.consensus_state.participants)

    @property
    def quorum_threshold(self) -> float:
        """Weighted quorum threshold: 2W/3."""
        return (2.0 * self.total_weight) / 3.0

    def weight_of(self, participant: str) -> float:
        """Get weight of a participant (default 1.0)."""
        return self.weights.get(participant, 1.0)

    def weight_of_set(self, participants: set[str]) -> float:
        """Get total weight of a set of participants."""
        return sum(self.weights.get(pid, 1.0) for pid in participants)

    def is_quorum(self, participants: set[str]) -> bool:
        """Check if a set of participants forms a weighted quorum."""
        return self.weight_of_set(participants) >= self.quorum_threshold

    def decided_weight(self) -> float:
        """Total weight of participants that have decided."""
        return self.weight_of_set(set(self.consensus_state.decisions.keys()))


class WeightedAgreement(FormalProperty[Any]):
    """Weighted Agreement: Correct processes with sufficient weight agree.

    For weighted quorum systems, agreement is enforced among processes
    whose combined weight exceeds the quorum threshold.
    """

    def __init__(self, min_agreement_weight: float | None = None):
        """Initialize WeightedAgreement.

        Args:
            min_agreement_weight: Minimum combined weight for agreement.
                If None, uses 2W/3 threshold.
        """
        super().__init__("WeightedAgreement", PropertyType.SAFETY)
        self._min_weight = min_agreement_weight

    def check(self, state: ConsensusState[Any], **kwargs: Any) -> PropertyResult:
        """Check weighted agreement property.

        Accepts WeightedConsensusState via kwargs['weighted_state']
        or wraps a regular ConsensusState with unit weights.
        """
        weighted = kwargs.get("weighted_state")
        if weighted is None:
            weighted = WeightedConsensusState(
                consensus_state=state,
                weights={pid: 1.0 for pid in state.participants},
            )

        threshold = self._min_weight or weighted.quorum_threshold
        decisions = weighted.consensus_state.decisions

        if len(decisions) < 2:
            return PropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=PropertyStatus.SATISFIED,
                message="Weighted agreement trivially holds (< 2 decisions)",
            )

        # Group by decision value
        value_groups: dict[str, set[str]] = {}
        for pid, value in decisions.items():
            key = str(value)
            if key not in value_groups:
                value_groups[key] = set()
            value_groups[key].add(pid)

        # Check if any two groups both have quorum weight
        quorum_groups = [
            (val, group)
            for val, group in value_groups.items()
            if weighted.weight_of_set(group) >= threshold
        ]

        if len(quorum_groups) > 1:
            return PropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=PropertyStatus.VIOLATED,
                message=(
                    f"Weighted agreement violated: {len(quorum_groups)} "
                    f"groups with quorum weight decided differently"
                ),
                counterexample={
                    "groups": {
                        val: {
                            "participants": list(group),
                            "weight": weighted.weight_of_set(group),
                        }
                        for val, group in quorum_groups
                    },
                    "threshold": threshold,
                },
            )

        return PropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=PropertyStatus.SATISFIED,
            message="Weighted agreement holds",
        )


class WeightedValidity(FormalProperty[Any]):
    """Weighted Validity: Decided value was proposed by a weighted quorum.

    Strengthens standard validity — the decided value must have been
    proposed by a set of processes with sufficient combined weight.
    """

    def __init__(self, min_proposal_weight: float | None = None):
        """Initialize WeightedValidity.

        Args:
            min_proposal_weight: Minimum weight of proposers for validity.
                If None, any single proposer suffices (standard validity).
        """
        super().__init__("WeightedValidity", PropertyType.SAFETY)
        self._min_weight = min_proposal_weight

    def check(self, state: ConsensusState[Any], **kwargs: Any) -> PropertyResult:
        """Check weighted validity."""
        weighted = kwargs.get("weighted_state")
        if weighted is None:
            weighted = WeightedConsensusState(
                consensus_state=state,
                weights={pid: 1.0 for pid in state.participants},
            )

        decisions = weighted.consensus_state.decisions
        proposals = weighted.consensus_state.proposed_values

        for pid, decision in decisions.items():
            # Find which processes proposed this value
            proposers = {p for p, v in proposals.items() if str(v) == str(decision)}

            if not proposers:
                return PropertyResult(
                    property_name=self.name,
                    property_type=self.property_type,
                    status=PropertyStatus.VIOLATED,
                    message=f"Process {pid} decided value not in any proposal",
                    counterexample={
                        "process": pid,
                        "decision": decision,
                    },
                )

            if self._min_weight is not None:
                proposer_weight = weighted.weight_of_set(proposers)
                if proposer_weight < self._min_weight:
                    return PropertyResult(
                        property_name=self.name,
                        property_type=self.property_type,
                        status=PropertyStatus.VIOLATED,
                        message=(
                            f"Decision supported by weight {proposer_weight:.2f} "
                            f"< required {self._min_weight:.2f}"
                        ),
                        counterexample={
                            "proposers": list(proposers),
                            "weight": proposer_weight,
                            "required": self._min_weight,
                        },
                    )

        return PropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=PropertyStatus.SATISFIED,
            message="Weighted validity holds",
        )


class WeightedQuorum(FormalProperty[Any]):
    """Weighted Quorum intersection property.

    Verifies that any two quorums have sufficient intersection weight
    to guarantee agreement. For BFT: any two quorums must share
    honest weight > W/3.
    """

    def __init__(self, byzantine_weight: float = 0.0):
        """Initialize WeightedQuorum.

        Args:
            byzantine_weight: Maximum total weight of Byzantine processes.
        """
        super().__init__("WeightedQuorum", PropertyType.INVARIANT)
        self._byzantine_weight = byzantine_weight

    def check(self, state: ConsensusState[Any], **kwargs: Any) -> PropertyResult:
        """Check quorum intersection property.

        Verifies: For any two quorums Q1, Q2:
        weight(Q1 ∩ Q2) > byzantine_weight
        """
        weighted = kwargs.get("weighted_state")
        if weighted is None:
            weighted = WeightedConsensusState(
                consensus_state=state,
                weights={pid: 1.0 for pid in state.participants},
            )

        total_w = weighted.total_weight
        quorum_thresh = weighted.quorum_threshold

        # For BFT correctness: 2 * quorum_thresh - total_w > byzantine_weight
        # This ensures any two quorums overlap by more than the Byzantine weight
        min_intersection_weight = 2 * quorum_thresh - total_w

        if min_intersection_weight > self._byzantine_weight:
            return PropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=PropertyStatus.SATISFIED,
                message=(
                    f"Quorum intersection weight ({min_intersection_weight:.2f}) > "
                    f"Byzantine weight ({self._byzantine_weight:.2f})"
                ),
            )

        return PropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=PropertyStatus.VIOLATED,
            message=(
                f"Insufficient quorum intersection: "
                f"{min_intersection_weight:.2f} <= "
                f"{self._byzantine_weight:.2f}"
            ),
            counterexample={
                "total_weight": total_w,
                "quorum_threshold": quorum_thresh,
                "min_intersection": min_intersection_weight,
                "byzantine_weight": self._byzantine_weight,
            },
        )


def select_weighted_leader(
    weighted_state: WeightedConsensusState,
    round_number: int,
) -> str:
    """Select leader based on weight-proportional rotation.

    Higher-weight agents get selected more frequently.

    Args:
        weighted_state: Weighted consensus state
        round_number: Current round number

    Returns:
        ID of selected leader
    """
    participants = sorted(weighted_state.consensus_state.participants)
    if not participants:
        msg = "No participants for leader selection"
        raise ValueError(msg)

    # Build cumulative weight array
    cumulative: list[tuple[str, float]] = []
    running = 0.0
    for pid in participants:
        running += weighted_state.weight_of(pid)
        cumulative.append((pid, running))

    total = running
    if total == 0:
        # Fallback to round-robin
        return participants[round_number % len(participants)]

    # Deterministic selection based on round number
    # Use modular arithmetic to pick position in weight space
    position = (round_number * 7919) % int(total * 1000) / 1000.0  # prime for spread

    for pid, cum_weight in cumulative:
        if position < cum_weight:
            return pid

    return cumulative[-1][0]


__all__ = [
    "WeightedConsensusState",
    "WeightedAgreement",
    "WeightedValidity",
    "WeightedQuorum",
    "select_weighted_leader",
]
