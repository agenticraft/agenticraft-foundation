"""Session type checker with runtime verification.

This module provides type checking for MPST sessions, integrating with
the InvariantRegistry for runtime verification of session properties.

Key Features:
- Well-formedness checking for global types
- Session conformance checking against local types
- Runtime monitoring with StateTransitionMonitor
- Integration with InvariantRegistry for property verification

References:
- Honda, Yoshida, Carbone (2008) - Multiparty Session Types
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agenticraft_foundation.mpst.global_types import (
    ChoiceType,
    MessageType,
    ParallelType,
    RecursionType,
    VariableType,
)
from agenticraft_foundation.mpst.local_types import (
    BranchType,
    LocalEndType,
    LocalRecursionType,
    Projector,
    ReceiveType,
    SelectType,
    SendType,
)
from agenticraft_foundation.mpst.types import (
    MessagePayload,
    ParticipantId,
    SessionType,
    SessionViolation,
    TypeCheckError,
)
from agenticraft_foundation.verification.invariant_checker import (
    InvariantRegistry,
    StateTransitionMonitor,
    ViolationSeverity,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Well-Formedness Checking
# =============================================================================


@dataclass
class WellFormednessResult:
    """Result of well-formedness checking.

    Attributes:
        is_well_formed: Whether the type is well-formed
        errors: List of errors found
        warnings: List of warnings
        participants: Set of participants in the type
    """

    is_well_formed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    participants: set[ParticipantId] = field(default_factory=set)


class WellFormednessChecker:
    """Checks well-formedness of global session types.

    A global type G is well-formed if:
    1. All participants are reachable from the initial state
    2. No orphan messages (every send has a matching receive)
    3. Choice consistency (sender/receiver are distinct)
    4. Recursion is guarded (no unguarded recursive calls)
    5. Projection is defined for all participants
    """

    def __init__(self) -> None:
        self._projector = Projector(strict=False)
        self._rec_vars: set[str] = set()

    def check(self, global_type: SessionType) -> WellFormednessResult:
        """Check if a global type is well-formed.

        Args:
            global_type: Global session type to check

        Returns:
            WellFormednessResult with details
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Get participants
        participants = global_type.participants()
        if not participants:
            errors.append("Global type has no participants")
            return WellFormednessResult(
                is_well_formed=False,
                errors=errors,
                participants=participants,
            )

        # Check projection is defined for all participants
        for p in participants:
            local = self._projector.project(global_type, p)
            if local is None:
                errors.append(f"Projection undefined for participant {p}")

        # Check structure
        self._check_structure(global_type, errors, warnings)

        # Check recursion
        self._check_recursion(global_type, errors, warnings, set())

        return WellFormednessResult(
            is_well_formed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            participants=participants,
        )

    def _check_structure(
        self,
        t: SessionType,
        errors: list[str],
        warnings: list[str],
    ) -> None:
        """Check structural properties."""
        if isinstance(t, MessageType):
            if t.sender == t.receiver:
                errors.append(f"Self-communication: {t.sender} → {t.sender}")
            self._check_structure(t.continuation, errors, warnings)

        elif isinstance(t, ChoiceType):
            if t.sender == t.receiver:
                errors.append(f"Self-communication in choice: {t.sender}")
            if len(t.branches) == 0:
                errors.append("Empty choice branches")
            for _label, branch in t.branches.items():
                self._check_structure(branch, errors, warnings)

        elif isinstance(t, RecursionType):
            self._check_structure(t.body, errors, warnings)

        elif isinstance(t, ParallelType):
            # Check for participant overlap (warning)
            left_p = t.left.participants()
            right_p = t.right.participants()
            overlap = left_p & right_p
            if overlap:
                warnings.append(f"Parallel composition has overlapping participants: {overlap}")
            self._check_structure(t.left, errors, warnings)
            self._check_structure(t.right, errors, warnings)

    def _check_recursion(
        self,
        t: SessionType,
        errors: list[str],
        warnings: list[str],
        bound_vars: set[str],
    ) -> None:
        """Check recursion is well-formed (guarded)."""
        if isinstance(t, RecursionType):
            if t.variable in bound_vars:
                errors.append(f"Shadowed recursion variable: {t.variable}")
            new_bound = bound_vars | {t.variable}
            # Check body is guarded (not immediately a variable)
            if isinstance(t.body, VariableType) and t.body.name == t.variable:
                errors.append(f"Unguarded recursion: μ{t.variable}.{t.variable}")
            self._check_recursion(t.body, errors, warnings, new_bound)

        elif isinstance(t, VariableType):
            if t.name not in bound_vars:
                errors.append(f"Unbound recursion variable: {t.name}")

        elif isinstance(t, MessageType):
            self._check_recursion(t.continuation, errors, warnings, bound_vars)

        elif isinstance(t, ChoiceType):
            for branch in t.branches.values():
                self._check_recursion(branch, errors, warnings, bound_vars)

        elif isinstance(t, ParallelType):
            self._check_recursion(t.left, errors, warnings, bound_vars)
            self._check_recursion(t.right, errors, warnings, bound_vars)


# =============================================================================
# Session Monitor
# =============================================================================


@dataclass
class SessionMonitor:
    """Monitors a session's conformance to its local type.

    Tracks message exchanges and verifies they conform to the
    expected local type for each participant.

    Integrates with StateTransitionMonitor for transition tracking.
    """

    participant: ParticipantId
    local_type: SessionType
    session_id: str = ""
    strict: bool = True

    def __post_init__(self) -> None:
        self._current_type: SessionType = self.local_type
        self._state_monitor = StateTransitionMonitor(
            name=f"mpst_session_{self.participant}_{self.session_id}",
        )
        self._violations: list[SessionViolation] = []
        self._message_count = 0

    def on_send(
        self,
        receiver: ParticipantId,
        payload: MessagePayload,
    ) -> bool:
        """Handle a send action.

        Args:
            receiver: Target participant
            payload: Message payload

        Returns:
            True if send conforms to local type

        Raises:
            TypeCheckError: If strict and send doesn't conform
        """
        self._message_count += 1

        # Unfold if needed
        current = self._unfold_type(self._current_type)

        if isinstance(current, SendType):
            if current.receiver == receiver and current.payload.label == payload.label:
                # Conformant: advance to continuation
                self._current_type = current.continuation
                self._state_monitor.transition(f"send_{payload.label}")
                return True
            else:
                violation = SessionViolation(
                    violation_type="send_mismatch",
                    message=f"Expected send to {current.receiver}({current.payload.label}), "
                    f"got send to {receiver}({payload.label})",
                    participant=self.participant,
                    expected_action=f"!{current.receiver}({current.payload.label})",
                    actual_action=f"!{receiver}({payload.label})",
                )
                return self._handle_violation(violation)

        elif isinstance(current, SelectType):
            if current.receiver == receiver and payload.label in current.branches:
                # Conformant: select branch
                self._current_type = current.branches[payload.label]
                self._state_monitor.transition(f"select_{payload.label}")
                return True
            else:
                violation = SessionViolation(
                    violation_type="select_mismatch",
                    message=f"Invalid selection: {payload.label} to {receiver}",
                    participant=self.participant,
                    expected_action=f"⊕{current.receiver}{{{list(current.branches.keys())}}}",
                    actual_action=f"!{receiver}({payload.label})",
                )
                return self._handle_violation(violation)

        else:
            violation = SessionViolation(
                violation_type="unexpected_send",
                message=f"Unexpected send: expected {type(current).__name__}",
                participant=self.participant,
                expected_action=str(current),
                actual_action=f"!{receiver}({payload.label})",
            )
            return self._handle_violation(violation)

    def on_receive(
        self,
        sender: ParticipantId,
        payload: MessagePayload,
    ) -> bool:
        """Handle a receive action.

        Args:
            sender: Source participant
            payload: Message payload

        Returns:
            True if receive conforms to local type
        """
        self._message_count += 1

        current = self._unfold_type(self._current_type)

        if isinstance(current, ReceiveType):
            if current.sender == sender and current.payload.label == payload.label:
                self._current_type = current.continuation
                self._state_monitor.transition(f"recv_{payload.label}")
                return True
            else:
                violation = SessionViolation(
                    violation_type="receive_mismatch",
                    message=f"Expected receive from {current.sender}({current.payload.label}), "
                    f"got receive from {sender}({payload.label})",
                    participant=self.participant,
                    expected_action=f"?{current.sender}({current.payload.label})",
                    actual_action=f"?{sender}({payload.label})",
                )
                return self._handle_violation(violation)

        elif isinstance(current, BranchType):
            if current.sender == sender and payload.label in current.branches:
                self._current_type = current.branches[payload.label]
                self._state_monitor.transition(f"branch_{payload.label}")
                return True
            else:
                violation = SessionViolation(
                    violation_type="branch_mismatch",
                    message=f"Invalid branch: {payload.label} from {sender}",
                    participant=self.participant,
                    expected_action=f"&{current.sender}{{{list(current.branches.keys())}}}",
                    actual_action=f"?{sender}({payload.label})",
                )
                return self._handle_violation(violation)

        else:
            violation = SessionViolation(
                violation_type="unexpected_receive",
                message=f"Unexpected receive: expected {type(current).__name__}",
                participant=self.participant,
                expected_action=str(current),
                actual_action=f"?{sender}({payload.label})",
            )
            return self._handle_violation(violation)

    def _unfold_type(self, t: SessionType) -> SessionType:
        """Unfold recursion types."""
        while isinstance(t, LocalRecursionType):
            t = t.unfold({})
        return t

    def _handle_violation(self, violation: SessionViolation) -> bool:
        """Handle a type violation."""
        self._violations.append(violation)
        logger.warning(f"Session type violation: {violation.message}")

        if self.strict:
            raise TypeCheckError(
                message=violation.message,
                location=f"participant={self.participant}, msg={self._message_count}",
            )
        return False

    def is_complete(self) -> bool:
        """Check if session has completed successfully."""
        current = self._unfold_type(self._current_type)
        return isinstance(current, LocalEndType)

    def can_send(self) -> bool:
        """Check if next action is a send."""
        current = self._unfold_type(self._current_type)
        return isinstance(current, (SendType, SelectType))

    def can_receive(self) -> bool:
        """Check if next action is a receive."""
        current = self._unfold_type(self._current_type)
        return isinstance(current, (ReceiveType, BranchType))

    def get_violations(self) -> list[SessionViolation]:
        """Get all recorded violations."""
        return self._violations

    def get_state(self) -> dict[str, Any]:
        """Get current monitor state."""
        return {
            "participant": self.participant,
            "session_id": self.session_id,
            "current_type": str(self._current_type),
            "message_count": self._message_count,
            "violation_count": len(self._violations),
            "is_complete": self.is_complete(),
            "can_send": self.can_send(),
            "can_receive": self.can_receive(),
        }


# =============================================================================
# MPST Invariant Registry
# =============================================================================


class MPSTInvariantRegistry:
    """Registry of MPST-specific invariants.

    Provides standard session type invariants for runtime verification.
    """

    def __init__(self, registry: InvariantRegistry | None = None):
        """Initialize with optional parent registry.

        Args:
            registry: Parent InvariantRegistry to use
        """
        self.registry = registry or InvariantRegistry("mpst")
        self._register_invariants()

    def _register_invariants(self) -> None:
        """Register standard MPST invariants."""

        # Deadlock freedom invariant
        self.registry.register(
            "mpst_no_deadlock",
            self._check_no_deadlock,
            severity=ViolationSeverity.FATAL,
            message="Session deadlock detected",
        )

        # Progress guarantee
        self.registry.register(
            "mpst_progress",
            self._check_progress,
            severity=ViolationSeverity.ERROR,
            message="Session cannot make progress",
        )

        # Type conformance
        self.registry.register(
            "mpst_conformance",
            self._check_conformance,
            severity=ViolationSeverity.ERROR,
            message="Session type violation",
        )

    def _check_no_deadlock(self, monitors: dict[ParticipantId, SessionMonitor]) -> bool:
        """Check that session is not deadlocked.

        A session is deadlocked if:
        - No participant can make progress
        - Not all participants are complete
        """
        if not monitors:
            return True

        all_complete = all(m.is_complete() for m in monitors.values())
        if all_complete:
            return True

        # Check if someone can send and someone can receive
        senders = [p for p, m in monitors.items() if m.can_send()]
        receivers = [p for p, m in monitors.items() if m.can_receive()]

        # Simple check: if there are senders but no receivers, potential deadlock
        if senders and not receivers:
            return False
        if receivers and not senders:
            return False

        return True

    def _check_progress(self, monitors: dict[ParticipantId, SessionMonitor]) -> bool:
        """Check that session can make progress."""
        if not monitors:
            return True

        # At least one participant should be able to act
        can_act = any(m.can_send() or m.can_receive() or m.is_complete() for m in monitors.values())
        return can_act

    def _check_conformance(self, monitors: dict[ParticipantId, SessionMonitor]) -> bool:
        """Check all monitors have no violations."""
        return all(len(m.get_violations()) == 0 for m in monitors.values())

    def check_all(
        self,
        monitors: dict[ParticipantId, SessionMonitor],
    ) -> list[str]:
        """Check all MPST invariants.

        Args:
            monitors: Dictionary of session monitors

        Returns:
            List of violated invariant names
        """
        return self.registry.check_all(monitors)


# =============================================================================
# Session Type Checker
# =============================================================================


class SessionTypeChecker:
    """Complete session type checker with runtime monitoring.

    Provides:
    - Well-formedness checking of global types
    - Local type projection
    - Runtime session monitoring
    - Invariant verification
    """

    def __init__(self, registry: InvariantRegistry | None = None):
        """Initialize type checker.

        Args:
            registry: Optional InvariantRegistry for invariants
        """
        self._well_formedness = WellFormednessChecker()
        self._projector = Projector(strict=True)
        self._mpst_registry = MPSTInvariantRegistry(registry)
        self._active_sessions: dict[str, dict[ParticipantId, SessionMonitor]] = {}

    def check_well_formed(self, global_type: SessionType) -> WellFormednessResult:
        """Check if a global type is well-formed.

        Args:
            global_type: Global session type

        Returns:
            WellFormednessResult with details
        """
        return self._well_formedness.check(global_type)

    def project(
        self,
        global_type: SessionType,
        participant: ParticipantId,
    ) -> SessionType:
        """Project global type to local type.

        Args:
            global_type: Global session type
            participant: Participant to project for

        Returns:
            Local session type
        """
        result = self._projector.project(global_type, participant)
        if result is None:
            raise TypeCheckError(
                message=f"Projection undefined for {participant}",
                location="projection",
            )
        return result

    def create_session(
        self,
        session_id: str,
        global_type: SessionType,
        strict: bool = True,
    ) -> dict[ParticipantId, SessionMonitor]:
        """Create monitors for a new session.

        Args:
            session_id: Unique session identifier
            global_type: Global session type
            strict: If True, raise on type violations

        Returns:
            Dictionary of monitors per participant
        """
        # Check well-formedness
        result = self.check_well_formed(global_type)
        if not result.is_well_formed:
            raise TypeCheckError(
                message=f"Global type not well-formed: {result.errors}",
                location="session_creation",
            )

        # Create monitors
        monitors = {}
        for participant in result.participants:
            local_type = self.project(global_type, participant)
            monitors[participant] = SessionMonitor(
                participant=participant,
                local_type=local_type,
                session_id=session_id,
                strict=strict,
            )

        self._active_sessions[session_id] = monitors
        return monitors

    def get_session(
        self,
        session_id: str,
    ) -> dict[ParticipantId, SessionMonitor] | None:
        """Get monitors for an active session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary of monitors or None if not found
        """
        return self._active_sessions.get(session_id)

    def close_session(self, session_id: str) -> bool:
        """Close and remove a session.

        Args:
            session_id: Session to close

        Returns:
            True if session was active and closed
        """
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            return True
        return False

    def check_invariants(self, session_id: str) -> list[str]:
        """Check MPST invariants for a session.

        Args:
            session_id: Session to check

        Returns:
            List of violated invariant names
        """
        monitors = self._active_sessions.get(session_id, {})
        return self._mpst_registry.check_all(monitors)

    def get_statistics(self) -> dict[str, Any]:
        """Get checker statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "active_sessions": len(self._active_sessions),
            "invariant_stats": self._mpst_registry.registry.stats(),
        }


__all__ = [
    # Well-formedness
    "WellFormednessResult",
    "WellFormednessChecker",
    # Monitoring
    "SessionMonitor",
    "MPSTInvariantRegistry",
    # Type checker
    "SessionTypeChecker",
]
