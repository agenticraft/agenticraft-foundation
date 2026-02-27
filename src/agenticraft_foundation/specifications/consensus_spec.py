"""Formal specifications for consensus protocols.

This module provides executable formal specifications for verifying
consensus protocol correctness, based on distributed systems theory.

Key properties verified:
- Safety: Agreement, Validity, Integrity
- Liveness: Termination

References:
- Fischer, Lynch, Paterson (1985) - Impossibility of consensus
- Castro & Liskov (1999) - PBFT
- Lamport (1998) - Paxos Made Simple
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class PropertyType(str, Enum):
    """Types of formal properties."""

    SAFETY = "safety"  # Nothing bad ever happens
    LIVENESS = "liveness"  # Something good eventually happens
    INVARIANT = "invariant"  # Always true
    EVENTUAL = "eventual"  # Eventually true


class PropertyStatus(str, Enum):
    """Status of property verification."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


@dataclass
class PropertyResult:
    """Result of property verification.

    Attributes:
        property_name: Name of the property checked
        property_type: Type of property (safety, liveness, etc.)
        status: Verification status
        message: Human-readable message
        counterexample: Counterexample if violated
        trace: Execution trace leading to violation
        timestamp: When verification was performed
    """

    property_name: str
    property_type: PropertyType
    status: PropertyStatus
    message: str = ""
    counterexample: Any | None = None
    trace: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def is_satisfied(self) -> bool:
        """Check if property was satisfied."""
        return self.status == PropertyStatus.SATISFIED


@dataclass
class ConsensusState(Generic[T]):
    """State of a consensus instance.

    Attributes:
        instance_id: Unique identifier for this consensus instance
        participants: Set of participant IDs
        proposed_values: Values proposed by each participant
        decisions: Final decisions by each participant
        messages: Messages exchanged
        round: Current round number
        is_terminated: Whether consensus has terminated
        metadata: Additional state metadata
    """

    instance_id: str
    participants: set[str]
    proposed_values: dict[str, T] = field(default_factory=dict)
    decisions: dict[str, T] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    round: int = 0
    is_terminated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class FormalProperty(ABC, Generic[T]):
    """Base class for formal properties.

    Properties are executable specifications that can be checked
    against consensus states.
    """

    def __init__(self, name: str, property_type: PropertyType):
        """Initialize property.

        Args:
            name: Property name
            property_type: Type of property
        """
        self.name = name
        self.property_type = property_type

    @abstractmethod
    def check(self, state: ConsensusState[T]) -> PropertyResult:
        """Check if property holds for given state.

        Args:
            state: Consensus state to check

        Returns:
            PropertyResult with verification outcome
        """
        pass


class Agreement(FormalProperty[Any]):
    """Agreement property: No two correct processes decide differently.

    Formally: For all correct processes p and q,
    if p decides v and q decides v', then v = v'.
    """

    def __init__(self, correct_processes: set[str] | None = None):
        """Initialize Agreement property.

        Args:
            correct_processes: Set of correct (non-faulty) process IDs.
                If None, all participants are assumed correct.
        """
        super().__init__("Agreement", PropertyType.SAFETY)
        self.correct_processes = correct_processes

    def check(self, state: ConsensusState[Any]) -> PropertyResult:
        """Check Agreement property.

        Args:
            state: Consensus state to check

        Returns:
            PropertyResult indicating if Agreement holds
        """
        correct = self.correct_processes or state.participants

        # Get decisions from correct processes only
        correct_decisions = {pid: value for pid, value in state.decisions.items() if pid in correct}

        if len(correct_decisions) < 2:
            # Not enough decisions to check agreement
            return PropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=PropertyStatus.SATISFIED,
                message="Agreement trivially holds (< 2 decisions)",
            )

        # Check all decisions are equal
        unique_values = set()
        for value in correct_decisions.values():
            # Handle unhashable types
            try:
                unique_values.add(value)
            except TypeError:
                unique_values.add(str(value))

        if len(unique_values) == 1:
            return PropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=PropertyStatus.SATISFIED,
                message=f"All {len(correct_decisions)} correct processes agree",
            )
        else:
            return PropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=PropertyStatus.VIOLATED,
                message=(f"Agreement violated: {len(unique_values)} different decisions"),
                counterexample={
                    "decisions": correct_decisions,
                    "unique_values": list(unique_values),
                },
            )


class Validity(FormalProperty[Any]):
    """Validity property: Decided value was proposed by some process.

    Formally: If a correct process decides v, then v was proposed
    by some process.
    """

    def __init__(self, correct_processes: set[str] | None = None):
        """Initialize Validity property.

        Args:
            correct_processes: Set of correct process IDs
        """
        super().__init__("Validity", PropertyType.SAFETY)
        self.correct_processes = correct_processes

    def check(self, state: ConsensusState[Any]) -> PropertyResult:
        """Check Validity property.

        Args:
            state: Consensus state to check

        Returns:
            PropertyResult indicating if Validity holds
        """
        correct = self.correct_processes or state.participants

        # Get all proposed values
        proposed_set = set()
        for value in state.proposed_values.values():
            try:
                proposed_set.add(value)
            except TypeError:
                proposed_set.add(str(value))

        # Check each correct process's decision
        for pid, decision in state.decisions.items():
            if pid not in correct:
                continue

            try:
                decision_in_proposed = decision in proposed_set
            except TypeError:
                decision_in_proposed = str(decision) in proposed_set

            if not decision_in_proposed:
                return PropertyResult(
                    property_name=self.name,
                    property_type=self.property_type,
                    status=PropertyStatus.VIOLATED,
                    message=f"Process {pid} decided value not in proposals",
                    counterexample={
                        "process": pid,
                        "decision": decision,
                        "proposals": list(proposed_set),
                    },
                )

        return PropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=PropertyStatus.SATISFIED,
            message="All decisions are from proposed values",
        )


class Integrity(FormalProperty[Any]):
    """Integrity property: A process decides at most once.

    Formally: For all correct processes p, p decides at most once.
    This is implicitly tracked by the decision being final.
    """

    def __init__(self, decision_history: dict[str, list[Any]] | None = None):
        """Initialize Integrity property.

        Args:
            decision_history: Historical decisions per process
        """
        super().__init__("Integrity", PropertyType.SAFETY)
        self.decision_history = decision_history or {}

    def check(self, state: ConsensusState[Any]) -> PropertyResult:
        """Check Integrity property.

        Args:
            state: Consensus state to check

        Returns:
            PropertyResult indicating if Integrity holds
        """
        # Check if any process has decided multiple times
        for pid in state.participants:
            history = self.decision_history.get(pid, [])
            if pid in state.decisions:
                history = history + [state.decisions[pid]]

            if len(history) > 1:
                # Check if decisions differ
                unique = len({str(v) for v in history})
                if unique > 1:
                    return PropertyResult(
                        property_name=self.name,
                        property_type=self.property_type,
                        status=PropertyStatus.VIOLATED,
                        message=f"Process {pid} decided multiple different values",
                        counterexample={
                            "process": pid,
                            "decisions": history,
                        },
                    )

        return PropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=PropertyStatus.SATISFIED,
            message="All processes have at most one decision",
        )


class Termination(FormalProperty[Any]):
    """Termination property: Every correct process eventually decides.

    This is a liveness property that requires eventual progress.
    Formally: Every correct process eventually decides some value.
    """

    def __init__(
        self,
        correct_processes: set[str] | None = None,
        timeout_rounds: int = 100,
    ):
        """Initialize Termination property.

        Args:
            correct_processes: Set of correct process IDs
            timeout_rounds: Maximum rounds before timeout
        """
        super().__init__("Termination", PropertyType.LIVENESS)
        self.correct_processes = correct_processes
        self.timeout_rounds = timeout_rounds

    def check(self, state: ConsensusState[Any]) -> PropertyResult:
        """Check Termination property.

        Args:
            state: Consensus state to check

        Returns:
            PropertyResult indicating if Termination holds/progresses
        """
        correct = self.correct_processes or state.participants

        # Check if all correct processes have decided
        undecided = [pid for pid in correct if pid not in state.decisions]

        if not undecided:
            return PropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=PropertyStatus.SATISFIED,
                message="All correct processes have terminated",
            )

        # Check for timeout
        if state.round >= self.timeout_rounds:
            return PropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=PropertyStatus.TIMEOUT,
                message=(
                    f"Timeout after {state.round} rounds, "
                    f"{len(undecided)} processes still undecided"
                ),
                counterexample={"undecided": undecided, "round": state.round},
            )

        # Still in progress
        return PropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=PropertyStatus.UNKNOWN,
            message=(
                f"In progress: {len(undecided)} of {len(correct)} "
                f"processes undecided at round {state.round}"
            ),
        )


class ConsensusSpecification:
    """Complete specification for consensus protocols.

    Combines all consensus properties and provides verification API.
    """

    def __init__(
        self,
        correct_processes: set[str] | None = None,
        custom_properties: list[FormalProperty[Any]] | None = None,
    ):
        """Initialize specification.

        Args:
            correct_processes: Set of correct process IDs
            custom_properties: Additional custom properties to check
        """
        self.correct_processes = correct_processes
        self.properties: list[FormalProperty[Any]] = [
            Agreement(correct_processes),
            Validity(correct_processes),
            Integrity(),
            Termination(correct_processes),
        ]
        if custom_properties:
            self.properties.extend(custom_properties)

        self._decision_history: dict[str, list[Any]] = {}

    def verify(self, state: ConsensusState[Any]) -> list[PropertyResult]:
        """Verify all properties against state.

        Args:
            state: Consensus state to verify

        Returns:
            List of PropertyResults for all properties
        """
        # Update decision history for integrity check
        for pid, decision in state.decisions.items():
            if pid not in self._decision_history:
                self._decision_history[pid] = []
            if decision not in self._decision_history[pid]:
                self._decision_history[pid].append(decision)

        # Update integrity checker with history
        for prop in self.properties:
            if isinstance(prop, Integrity):
                prop.decision_history = self._decision_history

        return [prop.check(state) for prop in self.properties]

    def verify_safety(self, state: ConsensusState[Any]) -> list[PropertyResult]:
        """Verify only safety properties.

        Args:
            state: Consensus state to verify

        Returns:
            List of PropertyResults for safety properties
        """
        safety_props = [p for p in self.properties if p.property_type == PropertyType.SAFETY]
        return [prop.check(state) for prop in safety_props]

    def verify_liveness(self, state: ConsensusState[Any]) -> list[PropertyResult]:
        """Verify only liveness properties.

        Args:
            state: Consensus state to verify

        Returns:
            List of PropertyResults for liveness properties
        """
        liveness_props = [p for p in self.properties if p.property_type == PropertyType.LIVENESS]
        return [prop.check(state) for prop in liveness_props]

    def is_valid(self, state: ConsensusState[Any]) -> bool:
        """Check if state satisfies all safety properties.

        Args:
            state: Consensus state to check

        Returns:
            True if all safety properties satisfied
        """
        results = self.verify_safety(state)
        return all(r.is_satisfied() for r in results)

    def summary(self, state: ConsensusState[Any]) -> str:
        """Generate verification summary.

        Args:
            state: Consensus state to summarize

        Returns:
            Human-readable summary
        """
        results = self.verify(state)

        lines = [
            "Consensus Specification Verification",
            "=" * 40,
            f"Instance: {state.instance_id}",
            f"Participants: {len(state.participants)}",
            f"Decisions: {len(state.decisions)}",
            f"Round: {state.round}",
            "",
            "Property Results:",
        ]

        for result in results:
            status_icon = {
                PropertyStatus.SATISFIED: "✓",
                PropertyStatus.VIOLATED: "✗",
                PropertyStatus.UNKNOWN: "?",
                PropertyStatus.TIMEOUT: "⏱",
            }.get(result.status, "?")

            lines.append(f"  {status_icon} {result.property_name}: {result.message}")

            if result.counterexample:
                lines.append(f"      Counterexample: {result.counterexample}")

        # Overall status
        all_safety = all(
            r.is_satisfied() for r in results if r.property_type == PropertyType.SAFETY
        )
        lines.append("")
        lines.append(f"Safety: {'PASS' if all_safety else 'FAIL'}")

        return "\n".join(lines)


@dataclass
class InvariantChecker:
    """Runtime invariant checker for consensus protocols.

    Monitors state transitions and checks invariants are maintained.
    """

    invariants: list[FormalProperty[Any]] = field(default_factory=list)
    violations: list[PropertyResult] = field(default_factory=list)
    check_count: int = 0

    def add_invariant(self, invariant: FormalProperty[Any]) -> None:
        """Add an invariant to check.

        Args:
            invariant: Invariant property to monitor
        """
        self.invariants.append(invariant)

    def check(self, state: ConsensusState[Any]) -> bool:
        """Check all invariants against current state.

        Args:
            state: Current consensus state

        Returns:
            True if all invariants hold
        """
        self.check_count += 1
        all_satisfied = True

        for invariant in self.invariants:
            result = invariant.check(state)
            if not result.is_satisfied():
                self.violations.append(result)
                all_satisfied = False

        return all_satisfied

    def get_violations(self) -> list[PropertyResult]:
        """Get all recorded violations.

        Returns:
            List of violation results
        """
        return self.violations

    def clear(self) -> None:
        """Clear violation history."""
        self.violations = []
        self.check_count = 0


def create_byzantine_spec(
    n: int,
    f: int,
) -> ConsensusSpecification:
    """Create specification for Byzantine fault tolerant consensus.

    Requires n >= 3f + 1 for safety.

    Args:
        n: Total number of processes
        f: Maximum Byzantine failures

    Returns:
        ConsensusSpecification configured for BFT
    """
    if n < 3 * f + 1:
        raise ValueError(f"BFT requires n >= 3f+1. Got n={n}, f={f}")

    # In BFT, we don't know which processes are faulty
    # But we assume at most f are Byzantine
    return ConsensusSpecification()


def create_crash_spec(
    n: int,
    f: int,
) -> ConsensusSpecification:
    """Create specification for crash fault tolerant consensus.

    Requires n >= 2f + 1 for safety.

    Args:
        n: Total number of processes
        f: Maximum crash failures

    Returns:
        ConsensusSpecification configured for CFT
    """
    if n < 2 * f + 1:
        raise ValueError(f"CFT requires n >= 2f+1. Got n={n}, f={f}")

    return ConsensusSpecification()


def hash_state(state: ConsensusState[Any]) -> str:
    """Compute hash of consensus state for verification.

    Args:
        state: State to hash

    Returns:
        SHA256 hash of state
    """
    content = (
        f"{state.instance_id}:"
        f"{sorted(state.participants)}:"
        f"{sorted(state.proposed_values.items())}:"
        f"{sorted(state.decisions.items())}:"
        f"{state.round}"
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


__all__ = [
    "PropertyType",
    "PropertyStatus",
    "PropertyResult",
    "ConsensusState",
    "FormalProperty",
    "Agreement",
    "Validity",
    "Integrity",
    "Termination",
    "ConsensusSpecification",
    "InvariantChecker",
    "create_byzantine_spec",
    "create_crash_spec",
    "hash_state",
]
