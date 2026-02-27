"""Formal properties for Multiparty Session Types.

This module provides executable formal properties for verifying
MPST correctness, following the FormalProperty pattern.

Key Properties:
- Safety: Deadlock freedom, type preservation
- Liveness: Progress, session completion
- Well-formedness: Projection definedness, merge consistency

References:
- Honda, Yoshida, Carbone (2008) - Multiparty Session Types
- Deniélou, Yoshida (2012) - Multiparty Session Types Meet Communicating Automata
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticraft_foundation.mpst.checker import (
    SessionMonitor,
    WellFormednessChecker,
)
from agenticraft_foundation.mpst.local_types import (
    Projector,
)
from agenticraft_foundation.mpst.types import (
    ParticipantId,
    SessionContext,
    SessionType,
)


class MPSTPropertyType(str, Enum):
    """Types of MPST properties."""

    SAFETY = "safety"  # Nothing bad ever happens
    LIVENESS = "liveness"  # Something good eventually happens
    WELL_FORMEDNESS = "well_formedness"  # Structural correctness
    PROGRESS = "progress"  # System can make progress


class MPSTPropertyStatus(str, Enum):
    """Status of MPST property verification."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


@dataclass
class MPSTPropertyResult:
    """Result of MPST property verification.

    Attributes:
        property_name: Name of the property
        property_type: Type of property
        status: Verification status
        message: Human-readable message
        counterexample: Counterexample if violated
        participants: Participants involved
        timestamp: When verification was performed
    """

    property_name: str
    property_type: MPSTPropertyType
    status: MPSTPropertyStatus
    message: str = ""
    counterexample: Any | None = None
    participants: set[ParticipantId] = field(default_factory=set)
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    def is_satisfied(self) -> bool:
        """Check if property was satisfied."""
        return self.status == MPSTPropertyStatus.SATISFIED


class MPSTProperty(ABC):
    """Base class for MPST formal properties.

    Properties are executable specifications that can be checked
    against session types and runtime contexts.
    """

    def __init__(self, name: str, property_type: MPSTPropertyType):
        """Initialize property.

        Args:
            name: Property name
            property_type: Type of property
        """
        self.name = name
        self.property_type = property_type

    @abstractmethod
    def check(
        self,
        global_type: SessionType | None = None,
        context: SessionContext | None = None,
        monitors: dict[ParticipantId, SessionMonitor] | None = None,
    ) -> MPSTPropertyResult:
        """Check if property holds.

        Args:
            global_type: Optional global session type
            context: Optional runtime context
            monitors: Optional session monitors

        Returns:
            MPSTPropertyResult with verification outcome
        """
        pass


# =============================================================================
# Well-Formedness Properties
# =============================================================================


class ProjectionDefinedness(MPSTProperty):
    """Property: Projection is defined for all participants.

    A global type G is projectable if G ↓ p is defined for all
    p ∈ participants(G).
    """

    def __init__(self) -> None:
        super().__init__("ProjectionDefinedness", MPSTPropertyType.WELL_FORMEDNESS)
        self._projector = Projector(strict=False)

    def check(
        self,
        global_type: SessionType | None = None,
        context: SessionContext | None = None,
        monitors: dict[ParticipantId, SessionMonitor] | None = None,
    ) -> MPSTPropertyResult:
        if global_type is None:
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.UNKNOWN,
                message="No global type provided",
            )

        participants = global_type.participants()
        undefined = []

        for p in participants:
            local = self._projector.project(global_type, p)
            if local is None:
                undefined.append(p)

        if undefined:
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.VIOLATED,
                message=f"Projection undefined for: {undefined}",
                counterexample={"undefined_participants": undefined},
                participants=participants,
            )

        return MPSTPropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=MPSTPropertyStatus.SATISFIED,
            message=f"Projection defined for all {len(participants)} participants",
            participants=participants,
        )


class ChoiceConsistency(MPSTProperty):
    """Property: Choice sender and receiver are distinct.

    In a choice type p → q : {lᵢ: Gᵢ}, we require p ≠ q.
    """

    def __init__(self) -> None:
        super().__init__("ChoiceConsistency", MPSTPropertyType.WELL_FORMEDNESS)

    def check(
        self,
        global_type: SessionType | None = None,
        context: SessionContext | None = None,
        monitors: dict[ParticipantId, SessionMonitor] | None = None,
    ) -> MPSTPropertyResult:
        if global_type is None:
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.UNKNOWN,
                message="No global type provided",
            )

        checker = WellFormednessChecker()
        result = checker.check(global_type)

        if result.is_well_formed:
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.SATISFIED,
                message="All choices are consistent",
                participants=result.participants,
            )

        # Filter for choice-related errors
        choice_errors = [e for e in result.errors if "choice" in e.lower()]

        return MPSTPropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=MPSTPropertyStatus.VIOLATED,
            message=f"Choice consistency violated: {choice_errors}",
            counterexample={"errors": choice_errors},
            participants=result.participants,
        )


# =============================================================================
# Safety Properties
# =============================================================================


class DeadlockFreedom(MPSTProperty):
    """Property: Session is deadlock-free.

    A session is deadlock-free if either:
    - All participants have completed, or
    - At least one participant can make progress
    """

    def __init__(self) -> None:
        super().__init__("DeadlockFreedom", MPSTPropertyType.SAFETY)

    def check(
        self,
        global_type: SessionType | None = None,
        context: SessionContext | None = None,
        monitors: dict[ParticipantId, SessionMonitor] | None = None,
    ) -> MPSTPropertyResult:
        # For global type: check well-formedness (implies deadlock-freedom)
        if global_type is not None:
            checker = WellFormednessChecker()
            result = checker.check(global_type)

            if result.is_well_formed:
                return MPSTPropertyResult(
                    property_name=self.name,
                    property_type=self.property_type,
                    status=MPSTPropertyStatus.SATISFIED,
                    message="Well-formed global type is deadlock-free",
                    participants=result.participants,
                )

        # For runtime: check monitors
        if monitors:
            all_complete = all(m.is_complete() for m in monitors.values())
            if all_complete:
                return MPSTPropertyResult(
                    property_name=self.name,
                    property_type=self.property_type,
                    status=MPSTPropertyStatus.SATISFIED,
                    message="All participants completed",
                    participants=set(monitors.keys()),
                )

            # Check if progress is possible
            senders = [p for p, m in monitors.items() if m.can_send()]
            receivers = [p for p, m in monitors.items() if m.can_receive()]

            if not senders and not receivers:
                # No one can act and not everyone is complete
                return MPSTPropertyResult(
                    property_name=self.name,
                    property_type=self.property_type,
                    status=MPSTPropertyStatus.VIOLATED,
                    message="Session deadlocked: no participant can proceed",
                    counterexample={
                        "senders": senders,
                        "receivers": receivers,
                        "complete": [p for p, m in monitors.items() if m.is_complete()],
                    },
                    participants=set(monitors.keys()),
                )

            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.SATISFIED,
                message=f"Session can progress: {len(senders)} senders, {len(receivers)} receivers",
                participants=set(monitors.keys()),
            )

        return MPSTPropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=MPSTPropertyStatus.UNKNOWN,
            message="No type or monitors provided",
        )


class TypePreservation(MPSTProperty):
    """Property: Session execution preserves types.

    At each step, the remaining session conforms to its local type.
    """

    def __init__(self) -> None:
        super().__init__("TypePreservation", MPSTPropertyType.SAFETY)

    def check(
        self,
        global_type: SessionType | None = None,
        context: SessionContext | None = None,
        monitors: dict[ParticipantId, SessionMonitor] | None = None,
    ) -> MPSTPropertyResult:
        if not monitors:
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.UNKNOWN,
                message="No monitors provided",
            )

        # Check all monitors for violations
        violations = []
        for p, m in monitors.items():
            if m.get_violations():
                violations.extend([(p, v) for v in m.get_violations()])

        if violations:
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.VIOLATED,
                message=f"Type violations: {len(violations)} found",
                counterexample={"violations": violations},
                participants=set(monitors.keys()),
            )

        return MPSTPropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=MPSTPropertyStatus.SATISFIED,
            message="All messages conform to types",
            participants=set(monitors.keys()),
        )


# =============================================================================
# Liveness Properties
# =============================================================================


class Progress(MPSTProperty):
    """Property: Session can make progress.

    At least one participant can perform an action unless all are complete.
    """

    def __init__(self) -> None:
        super().__init__("Progress", MPSTPropertyType.LIVENESS)

    def check(
        self,
        global_type: SessionType | None = None,
        context: SessionContext | None = None,
        monitors: dict[ParticipantId, SessionMonitor] | None = None,
    ) -> MPSTPropertyResult:
        if not monitors:
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.UNKNOWN,
                message="No monitors provided",
            )

        complete = [p for p, m in monitors.items() if m.is_complete()]
        can_act = [p for p, m in monitors.items() if m.can_send() or m.can_receive()]

        if len(complete) == len(monitors):
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.SATISFIED,
                message="Session completed",
                participants=set(monitors.keys()),
            )

        if can_act:
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.SATISFIED,
                message=f"{len(can_act)} participants can act",
                participants=set(monitors.keys()),
            )

        return MPSTPropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=MPSTPropertyStatus.VIOLATED,
            message="No participant can make progress",
            counterexample={
                "complete": complete,
                "blocked": [p for p in monitors if p not in complete],
            },
            participants=set(monitors.keys()),
        )


class SessionCompletion(MPSTProperty):
    """Property: Session eventually completes.

    All participants eventually reach their end state.
    """

    def __init__(self, max_messages: int = 1000) -> None:
        super().__init__("SessionCompletion", MPSTPropertyType.LIVENESS)
        self.max_messages = max_messages

    def check(
        self,
        global_type: SessionType | None = None,
        context: SessionContext | None = None,
        monitors: dict[ParticipantId, SessionMonitor] | None = None,
    ) -> MPSTPropertyResult:
        if not monitors:
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.UNKNOWN,
                message="No monitors provided",
            )

        complete = [p for p, m in monitors.items() if m.is_complete()]
        incomplete = [p for p in monitors if p not in complete]

        if not incomplete:
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.SATISFIED,
                message="All participants completed",
                participants=set(monitors.keys()),
            )

        # Check if we've exceeded message limit (potential infinite loop)
        total_messages = sum(m._message_count for m in monitors.values())
        if total_messages > self.max_messages:
            return MPSTPropertyResult(
                property_name=self.name,
                property_type=self.property_type,
                status=MPSTPropertyStatus.TIMEOUT,
                message=f"Exceeded {self.max_messages} messages without completion",
                counterexample={
                    "total_messages": total_messages,
                    "incomplete": incomplete,
                },
                participants=set(monitors.keys()),
            )

        return MPSTPropertyResult(
            property_name=self.name,
            property_type=self.property_type,
            status=MPSTPropertyStatus.UNKNOWN,
            message=f"In progress: {len(incomplete)} participants incomplete",
            participants=set(monitors.keys()),
        )


# =============================================================================
# MPST Specification
# =============================================================================


class MPSTSpecification:
    """Complete specification for MPST verification.

    Combines all MPST properties for comprehensive verification.
    """

    def __init__(
        self,
        custom_properties: list[MPSTProperty] | None = None,
    ):
        """Initialize specification.

        Args:
            custom_properties: Additional custom properties
        """
        self.properties: list[MPSTProperty] = [
            # Well-formedness
            ProjectionDefinedness(),
            ChoiceConsistency(),
            # Safety
            DeadlockFreedom(),
            TypePreservation(),
            # Liveness
            Progress(),
            SessionCompletion(),
        ]
        if custom_properties:
            self.properties.extend(custom_properties)

    def verify(
        self,
        global_type: SessionType | None = None,
        context: SessionContext | None = None,
        monitors: dict[ParticipantId, SessionMonitor] | None = None,
    ) -> list[MPSTPropertyResult]:
        """Verify all properties.

        Args:
            global_type: Optional global session type
            context: Optional runtime context
            monitors: Optional session monitors

        Returns:
            List of property results
        """
        return [prop.check(global_type, context, monitors) for prop in self.properties]

    def verify_well_formedness(
        self,
        global_type: SessionType,
    ) -> list[MPSTPropertyResult]:
        """Verify only well-formedness properties."""
        wf_props = [
            p for p in self.properties if p.property_type == MPSTPropertyType.WELL_FORMEDNESS
        ]
        return [prop.check(global_type=global_type) for prop in wf_props]

    def verify_safety(
        self,
        monitors: dict[ParticipantId, SessionMonitor],
    ) -> list[MPSTPropertyResult]:
        """Verify only safety properties."""
        safety_props = [p for p in self.properties if p.property_type == MPSTPropertyType.SAFETY]
        return [prop.check(monitors=monitors) for prop in safety_props]

    def is_valid(
        self,
        global_type: SessionType | None = None,
        monitors: dict[ParticipantId, SessionMonitor] | None = None,
    ) -> bool:
        """Check if all safety properties are satisfied."""
        results = self.verify(global_type=global_type, monitors=monitors)
        safety_results = [
            r
            for r in results
            if r.property_type in (MPSTPropertyType.SAFETY, MPSTPropertyType.WELL_FORMEDNESS)
        ]
        return all(r.is_satisfied() for r in safety_results)

    def summary(
        self,
        global_type: SessionType | None = None,
        monitors: dict[ParticipantId, SessionMonitor] | None = None,
    ) -> str:
        """Generate verification summary."""
        results = self.verify(global_type=global_type, monitors=monitors)

        lines = [
            "MPST Specification Verification",
            "=" * 40,
            "",
            "Property Results:",
        ]

        for result in results:
            status_icon = {
                MPSTPropertyStatus.SATISFIED: "✓",
                MPSTPropertyStatus.VIOLATED: "✗",
                MPSTPropertyStatus.UNKNOWN: "?",
                MPSTPropertyStatus.TIMEOUT: "⏱",
            }.get(result.status, "?")

            lines.append(f"  {status_icon} {result.property_name}: {result.message}")

        # Overall status
        all_satisfied = all(r.is_satisfied() for r in results)
        lines.append("")
        lines.append(f"Overall: {'PASS' if all_satisfied else 'FAIL'}")

        return "\n".join(lines)


__all__ = [
    # Enums
    "MPSTPropertyType",
    "MPSTPropertyStatus",
    # Result
    "MPSTPropertyResult",
    # Base
    "MPSTProperty",
    # Well-formedness properties
    "ProjectionDefinedness",
    "ChoiceConsistency",
    # Safety properties
    "DeadlockFreedom",
    "TypePreservation",
    # Liveness properties
    "Progress",
    "SessionCompletion",
    # Specification
    "MPSTSpecification",
]
