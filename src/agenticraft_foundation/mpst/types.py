"""Core type definitions for Multiparty Session Types (MPST).

This module provides the foundational types for MPST-based verification
of multi-agent choreographies.

References:
- Honda, Yoshida, Carbone (2008) - Multiparty Session Types
- Scalas, Yoshida (2019) - Less is More: Multiparty Session Types Revisited

Key Theorem: Well-typed sessions are deadlock-free.
If all participants conform to their local projections,
the global protocol is deadlock-free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

T = TypeVar("T")


class ParticipantId(str):
    """Unique identifier for a session participant (agent)."""

    pass


class MessageLabel(str):
    """Label for a message type in session communication."""

    pass


class SessionState(str, Enum):
    """States of a session execution."""

    INITIAL = "initial"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    DEADLOCKED = "deadlocked"


class TypeKind(str, Enum):
    """Kinds of session types."""

    MESSAGE = "message"  # p → q : M
    CHOICE = "choice"  # p → q : {lᵢ: Gᵢ}
    RECURSION = "recursion"  # μX.G
    VARIABLE = "variable"  # X (recursion variable)
    END = "end"  # Session termination
    PARALLEL = "parallel"  # G₁ | G₂


@dataclass(frozen=True)
class MessagePayload:
    """Payload type for session messages.

    Attributes:
        label: Message label/type identifier
        payload_type: Type of the payload data
        schema: Optional JSON schema for validation
    """

    label: MessageLabel
    payload_type: str = "any"
    schema: dict[str, Any] | None = None

    def __hash__(self) -> int:
        return hash((self.label, self.payload_type))


@dataclass
class SessionMessage:
    """A message in a session.

    Attributes:
        sender: Sending participant
        receiver: Receiving participant
        payload: Message payload
        session_id: Session this message belongs to
        sequence: Sequence number in session
        metadata: Additional message metadata
    """

    sender: ParticipantId
    receiver: ParticipantId
    payload: MessagePayload
    session_id: str = ""
    sequence: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionContext:
    """Runtime context for session execution.

    Attributes:
        session_id: Unique session identifier
        participants: Set of participant IDs
        current_state: Current execution state
        message_history: History of messages exchanged
        local_states: Per-participant local state
        metadata: Additional context metadata
    """

    session_id: str
    participants: set[ParticipantId] = field(default_factory=set)
    current_state: SessionState = SessionState.INITIAL
    message_history: list[SessionMessage] = field(default_factory=list)
    local_states: dict[ParticipantId, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: SessionMessage) -> None:
        """Record a message in the session history."""
        message.session_id = self.session_id
        message.sequence = len(self.message_history)
        self.message_history.append(message)

    def get_participant_messages(self, participant: ParticipantId) -> list[SessionMessage]:
        """Get all messages involving a participant."""
        return [
            m for m in self.message_history if m.sender == participant or m.receiver == participant
        ]


class SessionType(ABC):
    """Abstract base for session types (both global and local).

    Session types describe the structure of multi-party communication.
    """

    @property
    @abstractmethod
    def kind(self) -> TypeKind:
        """The kind of session type."""
        ...

    @abstractmethod
    def participants(self) -> set[ParticipantId]:
        """Get all participants referenced in this type.

        Returns:
            Set of participant identifiers.
        """
        ...

    @abstractmethod
    def is_terminated(self) -> bool:
        """Check if this type represents termination."""
        ...

    @abstractmethod
    def unfold(self, bindings: dict[str, SessionType]) -> SessionType:
        """Unfold recursion variables with given bindings."""
        ...

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SessionType):
            return False
        return self._structural_eq(other)

    @abstractmethod
    def _structural_eq(self, other: SessionType) -> bool:
        """Structural equality check."""
        ...


@dataclass
class ProjectionError(Exception):
    """Error during type projection.

    Attributes:
        message: Error message
        participant: Participant being projected
        global_type: Global type being projected
        cause: Original exception if any
    """

    message: str
    participant: ParticipantId | None = None
    global_type: SessionType | None = None
    cause: Exception | None = None

    def __str__(self) -> str:
        parts = [self.message]
        if self.participant:
            parts.append(f"participant={self.participant}")
        return " ".join(parts)


@dataclass
class TypeCheckError(Exception):
    """Error during type checking.

    Attributes:
        message: Error message
        expected: Expected type
        actual: Actual type encountered
        location: Location in the session
    """

    message: str
    expected: SessionType | None = None
    actual: SessionType | None = None
    location: str = ""

    def __str__(self) -> str:
        return f"{self.message} at {self.location}" if self.location else self.message


@dataclass
class SessionViolation:
    """A violation of session type rules.

    Attributes:
        violation_type: Type of violation
        message: Detailed message
        participant: Participant involved
        expected_action: What was expected
        actual_action: What actually happened
        context: Additional context
    """

    violation_type: str
    message: str
    participant: ParticipantId | None = None
    expected_action: str = ""
    actual_action: str = ""
    context: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "ParticipantId",
    "MessageLabel",
    "SessionState",
    "TypeKind",
    "MessagePayload",
    "SessionMessage",
    "SessionContext",
    "SessionType",
    "ProjectionError",
    "TypeCheckError",
    "SessionViolation",
]
