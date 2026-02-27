"""Global session types for multi-party choreographies.

Global types describe the overall communication structure from a
bird's-eye view, specifying message exchanges between all participants.

Syntax:
- p → q : M.G        (Message: p sends M to q, then continue with G)
- p → q : {lᵢ: Gᵢ}  (Choice: p sends choice lᵢ to q, continue with Gᵢ)
- μX.G               (Recursion: recursive type bound to X)
- X                  (Variable: recursion variable reference)
- end                (End: successful termination)

References:
- Honda, Yoshida, Carbone (2008) - Multiparty Session Types
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agenticraft_foundation.mpst.types import (
    MessageLabel,
    MessagePayload,
    ParticipantId,
    SessionType,
    TypeKind,
)


@dataclass(frozen=True)
class EndType(SessionType):
    """End type: Successful session termination.

    Represents the end of a session with no further communication.
    Written as 'end' in session type notation.
    """

    @property
    def kind(self) -> TypeKind:
        return TypeKind.END

    def participants(self) -> set[ParticipantId]:
        return set()

    def is_terminated(self) -> bool:
        return True

    def unfold(self, bindings: dict[str, SessionType]) -> SessionType:
        return self

    def _structural_eq(self, other: SessionType) -> bool:
        return isinstance(other, EndType)

    def __repr__(self) -> str:
        return "end"


@dataclass
class MessageType(SessionType):
    """Message type: p → q : M.G

    Represents a single message from sender to receiver,
    followed by a continuation.

    Attributes:
        sender: Sending participant p
        receiver: Receiving participant q
        payload: Message payload M
        continuation: Continuation type G
    """

    sender: ParticipantId
    receiver: ParticipantId
    payload: MessagePayload
    continuation: SessionType = field(default_factory=EndType)

    def __post_init__(self) -> None:
        if self.sender == self.receiver:
            raise ValueError(f"Sender and receiver must differ: {self.sender} → {self.receiver}")

    @property
    def kind(self) -> TypeKind:
        return TypeKind.MESSAGE

    def participants(self) -> set[ParticipantId]:
        return {self.sender, self.receiver} | self.continuation.participants()

    def is_terminated(self) -> bool:
        return False

    def unfold(self, bindings: dict[str, SessionType]) -> SessionType:
        return MessageType(
            sender=self.sender,
            receiver=self.receiver,
            payload=self.payload,
            continuation=self.continuation.unfold(bindings),
        )

    def _structural_eq(self, other: SessionType) -> bool:
        if not isinstance(other, MessageType):
            return False
        return (
            self.sender == other.sender
            and self.receiver == other.receiver
            and self.payload == other.payload
            and self.continuation == other.continuation
        )

    def __repr__(self) -> str:
        return f"{self.sender} → {self.receiver} : {self.payload.label}.{self.continuation}"

    def __hash__(self) -> int:
        return hash((self.sender, self.receiver, self.payload, id(self.continuation)))


@dataclass
class ChoiceType(SessionType):
    """Choice type: p → q : {lᵢ: Gᵢ}

    Represents a branching choice where sender selects a label
    and both parties continue with the corresponding branch.

    Attributes:
        sender: Participant making the choice p
        receiver: Participant receiving the choice q
        branches: Mapping from labels to continuation types
    """

    sender: ParticipantId
    receiver: ParticipantId
    branches: dict[MessageLabel, SessionType] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sender == self.receiver:
            raise ValueError(f"Sender and receiver must differ: {self.sender} → {self.receiver}")
        if not self.branches:
            raise ValueError("Choice must have at least one branch")

    @property
    def kind(self) -> TypeKind:
        return TypeKind.CHOICE

    def participants(self) -> set[ParticipantId]:
        result = {self.sender, self.receiver}
        for branch in self.branches.values():
            result |= branch.participants()
        return result

    def is_terminated(self) -> bool:
        return False

    def unfold(self, bindings: dict[str, SessionType]) -> SessionType:
        return ChoiceType(
            sender=self.sender,
            receiver=self.receiver,
            branches={label: branch.unfold(bindings) for label, branch in self.branches.items()},
        )

    def _structural_eq(self, other: SessionType) -> bool:
        if not isinstance(other, ChoiceType):
            return False
        if self.sender != other.sender or self.receiver != other.receiver:
            return False
        if set(self.branches.keys()) != set(other.branches.keys()):
            return False
        return all(self.branches[label] == other.branches[label] for label in self.branches)

    def __repr__(self) -> str:
        branch_str = ", ".join(f"{label}: {branch}" for label, branch in self.branches.items())
        return f"{self.sender} → {self.receiver} : {{{branch_str}}}"

    def __hash__(self) -> int:
        return hash((self.sender, self.receiver, tuple(sorted(self.branches.keys()))))


@dataclass
class RecursionType(SessionType):
    """Recursion type: μX.G

    Represents a recursive session type where variable X
    is bound in the body G.

    Attributes:
        variable: Recursion variable name X
        body: Body of the recursion G
    """

    variable: str
    body: SessionType

    @property
    def kind(self) -> TypeKind:
        return TypeKind.RECURSION

    def participants(self) -> set[ParticipantId]:
        return self.body.participants()

    def is_terminated(self) -> bool:
        return False

    def unfold(self, bindings: dict[str, SessionType]) -> SessionType:
        # Add self to bindings and unfold body
        new_bindings = {**bindings, self.variable: self}
        return self.body.unfold(new_bindings)

    def unfold_once(self) -> SessionType:
        """Unfold one level of recursion.

        Returns the body with the variable replaced by self.
        """
        return self.body.unfold({self.variable: self})

    def _structural_eq(self, other: SessionType) -> bool:
        if not isinstance(other, RecursionType):
            return False
        # Alpha-equivalence: variables can have different names
        # if the structure is the same
        return self.variable == other.variable and self.body == other.body

    def __repr__(self) -> str:
        return f"μ{self.variable}.{self.body}"

    def __hash__(self) -> int:
        return hash((self.variable, id(self.body)))


@dataclass(frozen=True)
class VariableType(SessionType):
    """Variable type: X

    Represents a recursion variable reference.

    Attributes:
        name: Variable name
    """

    name: str

    @property
    def kind(self) -> TypeKind:
        return TypeKind.VARIABLE

    def participants(self) -> set[ParticipantId]:
        return set()

    def is_terminated(self) -> bool:
        return False

    def unfold(self, bindings: dict[str, SessionType]) -> SessionType:
        if self.name in bindings:
            return bindings[self.name]
        return self

    def _structural_eq(self, other: SessionType) -> bool:
        if not isinstance(other, VariableType):
            return False
        return self.name == other.name

    def __repr__(self) -> str:
        return self.name


@dataclass
class ParallelType(SessionType):
    """Parallel composition: G₁ | G₂

    Represents parallel execution of two independent sessions.
    Participants of G₁ and G₂ should be disjoint.

    Attributes:
        left: Left session type G₁
        right: Right session type G₂
    """

    left: SessionType
    right: SessionType

    def __post_init__(self) -> None:
        # Check for participant overlap (warning, not error)
        left_p = self.left.participants()
        right_p = self.right.participants()
        overlap = left_p & right_p
        if overlap:
            # Allow overlap for now, but could be stricter
            pass

    @property
    def kind(self) -> TypeKind:
        return TypeKind.PARALLEL

    def participants(self) -> set[ParticipantId]:
        return self.left.participants() | self.right.participants()

    def is_terminated(self) -> bool:
        return self.left.is_terminated() and self.right.is_terminated()

    def unfold(self, bindings: dict[str, SessionType]) -> SessionType:
        return ParallelType(
            left=self.left.unfold(bindings),
            right=self.right.unfold(bindings),
        )

    def _structural_eq(self, other: SessionType) -> bool:
        if not isinstance(other, ParallelType):
            return False
        return self.left == other.left and self.right == other.right

    def __repr__(self) -> str:
        return f"({self.left} | {self.right})"

    def __hash__(self) -> int:
        return hash((id(self.left), id(self.right)))


# Convenience constructors


def msg(
    sender: str | ParticipantId,
    receiver: str | ParticipantId,
    label: str | MessageLabel,
    continuation: SessionType | None = None,
    payload_type: str = "any",
) -> MessageType:
    """Convenience constructor for MessageType.

    Args:
        sender: Sending participant
        receiver: Receiving participant
        label: Message label
        continuation: Continuation type (default: EndType)
        payload_type: Type of payload

    Returns:
        MessageType instance

    Example:
        # Client → Server : Request. Server → Client : Response. end
        msg("client", "server", "request",
            msg("server", "client", "response"))
    """
    return MessageType(
        sender=ParticipantId(sender),
        receiver=ParticipantId(receiver),
        payload=MessagePayload(label=MessageLabel(label), payload_type=payload_type),
        continuation=continuation or EndType(),
    )


def choice(
    sender: str | ParticipantId,
    receiver: str | ParticipantId,
    branches: dict[str, SessionType],
) -> ChoiceType:
    """Convenience constructor for ChoiceType.

    Args:
        sender: Participant making the choice
        receiver: Participant receiving the choice
        branches: Mapping from label strings to continuation types

    Returns:
        ChoiceType instance

    Example:
        choice("client", "server", {
            "buy": msg("server", "client", "confirm"),
            "cancel": msg("server", "client", "cancelled"),
        })
    """
    return ChoiceType(
        sender=ParticipantId(sender),
        receiver=ParticipantId(receiver),
        branches={MessageLabel(k): v for k, v in branches.items()},
    )


def rec(variable: str, body: SessionType) -> RecursionType:
    """Convenience constructor for RecursionType.

    Args:
        variable: Recursion variable name
        body: Body of the recursion

    Returns:
        RecursionType instance

    Example:
        # μX. Client → Server : Request. Server → Client : Response. X
        rec("X", msg("client", "server", "request",
                    msg("server", "client", "response", var("X"))))
    """
    return RecursionType(variable=variable, body=body)


def var(name: str) -> VariableType:
    """Convenience constructor for VariableType.

    Args:
        name: Variable name

    Returns:
        VariableType instance
    """
    return VariableType(name=name)


def end() -> EndType:
    """Convenience constructor for EndType.

    Returns:
        EndType instance
    """
    return EndType()


def parallel(left: SessionType, right: SessionType) -> ParallelType:
    """Convenience constructor for ParallelType.

    Args:
        left: Left session type
        right: Right session type

    Returns:
        ParallelType instance
    """
    return ParallelType(left=left, right=right)


__all__ = [
    # Types
    "EndType",
    "MessageType",
    "ChoiceType",
    "RecursionType",
    "VariableType",
    "ParallelType",
    # Constructors
    "msg",
    "choice",
    "rec",
    "var",
    "end",
    "parallel",
]
