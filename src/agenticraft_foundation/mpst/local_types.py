"""Local session types and projection algorithm.

Local types describe the communication behavior from a single
participant's perspective. They are derived from global types
via projection.

Local Type Syntax:
- !q(M).L           (Send: send M to q, then continue with L)
- ?p(M).L           (Receive: receive M from p, then continue with L)
- ⊕q{lᵢ: Lᵢ}      (Select: send choice lᵢ to q)
- &p{lᵢ: Lᵢ}      (Branch: receive choice from p)
- μX.L              (Recursion)
- X                 (Variable)
- end               (End)

Projection Rules:
- (p → q : M.G) ↓ p = !q(M).(G ↓ p)      (sender projects to send)
- (p → q : M.G) ↓ q = ?p(M).(G ↓ q)      (receiver projects to receive)
- (p → q : M.G) ↓ r = G ↓ r              (other participants skip)

Key Theorem: If G ↓ p is defined for all p ∈ participants(G),
then the local types are consistent and the session is deadlock-free.

References:
- Honda, Yoshida, Carbone (2008) - Multiparty Session Types
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agenticraft_foundation.mpst.global_types import (
    ChoiceType,
    EndType,
    MessageType,
    ParallelType,
    RecursionType,
    VariableType,
)
from agenticraft_foundation.mpst.types import (
    MessageLabel,
    MessagePayload,
    ParticipantId,
    ProjectionError,
    SessionType,
    TypeKind,
)

# =============================================================================
# Local Session Types
# =============================================================================


@dataclass(frozen=True)
class LocalEndType(SessionType):
    """Local end type: Session termination from local perspective."""

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
        return isinstance(other, LocalEndType)

    def __repr__(self) -> str:
        return "end"


@dataclass
class SendType(SessionType):
    """Send type: !q(M).L

    Send message M to participant q, then continue with L.

    Attributes:
        receiver: Target participant q
        payload: Message payload M
        continuation: Continuation type L
    """

    receiver: ParticipantId
    payload: MessagePayload
    continuation: SessionType = field(default_factory=LocalEndType)

    @property
    def kind(self) -> TypeKind:
        return TypeKind.MESSAGE

    def participants(self) -> set[ParticipantId]:
        return {self.receiver} | self.continuation.participants()

    def is_terminated(self) -> bool:
        return False

    def unfold(self, bindings: dict[str, SessionType]) -> SessionType:
        return SendType(
            receiver=self.receiver,
            payload=self.payload,
            continuation=self.continuation.unfold(bindings),
        )

    def _structural_eq(self, other: SessionType) -> bool:
        if not isinstance(other, SendType):
            return False
        return (
            self.receiver == other.receiver
            and self.payload == other.payload
            and self.continuation == other.continuation
        )

    def __repr__(self) -> str:
        return f"!{self.receiver}({self.payload.label}).{self.continuation}"

    def __hash__(self) -> int:
        return hash((self.receiver, self.payload, id(self.continuation)))


@dataclass
class ReceiveType(SessionType):
    """Receive type: ?p(M).L

    Receive message M from participant p, then continue with L.

    Attributes:
        sender: Source participant p
        payload: Message payload M
        continuation: Continuation type L
    """

    sender: ParticipantId
    payload: MessagePayload
    continuation: SessionType = field(default_factory=LocalEndType)

    @property
    def kind(self) -> TypeKind:
        return TypeKind.MESSAGE

    def participants(self) -> set[ParticipantId]:
        return {self.sender} | self.continuation.participants()

    def is_terminated(self) -> bool:
        return False

    def unfold(self, bindings: dict[str, SessionType]) -> SessionType:
        return ReceiveType(
            sender=self.sender,
            payload=self.payload,
            continuation=self.continuation.unfold(bindings),
        )

    def _structural_eq(self, other: SessionType) -> bool:
        if not isinstance(other, ReceiveType):
            return False
        return (
            self.sender == other.sender
            and self.payload == other.payload
            and self.continuation == other.continuation
        )

    def __repr__(self) -> str:
        return f"?{self.sender}({self.payload.label}).{self.continuation}"

    def __hash__(self) -> int:
        return hash((self.sender, self.payload, id(self.continuation)))


@dataclass
class SelectType(SessionType):
    """Select type: ⊕q{lᵢ: Lᵢ}

    Send choice to participant q, selecting one of the labels.

    Attributes:
        receiver: Target participant q
        branches: Available choices with continuations
    """

    receiver: ParticipantId
    branches: dict[MessageLabel, SessionType] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("Select must have at least one branch")

    @property
    def kind(self) -> TypeKind:
        return TypeKind.CHOICE

    def participants(self) -> set[ParticipantId]:
        result = {self.receiver}
        for branch in self.branches.values():
            result |= branch.participants()
        return result

    def is_terminated(self) -> bool:
        return False

    def unfold(self, bindings: dict[str, SessionType]) -> SessionType:
        return SelectType(
            receiver=self.receiver,
            branches={label: branch.unfold(bindings) for label, branch in self.branches.items()},
        )

    def _structural_eq(self, other: SessionType) -> bool:
        if not isinstance(other, SelectType):
            return False
        if self.receiver != other.receiver:
            return False
        if set(self.branches.keys()) != set(other.branches.keys()):
            return False
        return all(self.branches[label] == other.branches[label] for label in self.branches)

    def __repr__(self) -> str:
        branch_str = ", ".join(f"{label}: {branch}" for label, branch in self.branches.items())
        return f"⊕{self.receiver}{{{branch_str}}}"

    def __hash__(self) -> int:
        return hash((self.receiver, tuple(sorted(self.branches.keys()))))


@dataclass
class BranchType(SessionType):
    """Branch type: &p{lᵢ: Lᵢ}

    Receive choice from participant p, branching on the label.

    Attributes:
        sender: Source participant p
        branches: Possible branches with continuations
    """

    sender: ParticipantId
    branches: dict[MessageLabel, SessionType] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("Branch must have at least one branch")

    @property
    def kind(self) -> TypeKind:
        return TypeKind.CHOICE

    def participants(self) -> set[ParticipantId]:
        result = {self.sender}
        for branch in self.branches.values():
            result |= branch.participants()
        return result

    def is_terminated(self) -> bool:
        return False

    def unfold(self, bindings: dict[str, SessionType]) -> SessionType:
        return BranchType(
            sender=self.sender,
            branches={label: branch.unfold(bindings) for label, branch in self.branches.items()},
        )

    def _structural_eq(self, other: SessionType) -> bool:
        if not isinstance(other, BranchType):
            return False
        if self.sender != other.sender:
            return False
        if set(self.branches.keys()) != set(other.branches.keys()):
            return False
        return all(self.branches[label] == other.branches[label] for label in self.branches)

    def __repr__(self) -> str:
        branch_str = ", ".join(f"{label}: {branch}" for label, branch in self.branches.items())
        return f"&{self.sender}{{{branch_str}}}"

    def __hash__(self) -> int:
        return hash((self.sender, tuple(sorted(self.branches.keys()))))


@dataclass
class LocalRecursionType(SessionType):
    """Local recursion type: μX.L

    Attributes:
        variable: Recursion variable name
        body: Body of the recursion
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
        new_bindings = {**bindings, self.variable: self}
        return self.body.unfold(new_bindings)

    def _structural_eq(self, other: SessionType) -> bool:
        if not isinstance(other, LocalRecursionType):
            return False
        return self.variable == other.variable and self.body == other.body

    def __repr__(self) -> str:
        return f"μ{self.variable}.{self.body}"

    def __hash__(self) -> int:
        return hash((self.variable, id(self.body)))


@dataclass(frozen=True)
class LocalVariableType(SessionType):
    """Local variable type: X

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
        if not isinstance(other, LocalVariableType):
            return False
        return self.name == other.name

    def __repr__(self) -> str:
        return self.name


# =============================================================================
# Projection Algorithm
# =============================================================================


class Projector:
    """Projects global session types to local types for each participant.

    The projection algorithm derives the local view of a session
    from the global choreography.

    Key Property: If projection is defined for all participants,
    the session is well-formed and deadlock-free.
    """

    def __init__(self, strict: bool = True):
        """Initialize projector.

        Args:
            strict: If True, raise errors on projection failure.
                   If False, return None on failure.
        """
        self.strict = strict
        self._recursion_stack: set[str] = set()

    def project(
        self,
        global_type: SessionType,
        participant: ParticipantId,
    ) -> SessionType | None:
        """Project global type to local type for participant.

        G ↓ p = local type for participant p

        Args:
            global_type: Global session type G
            participant: Participant to project to p

        Returns:
            Local session type (G ↓ p), or None if undefined

        Raises:
            ProjectionError: If projection fails and strict=True
        """
        try:
            return self._project(global_type, participant)
        except ProjectionError:
            if self.strict:
                raise
            return None

    def _project(
        self,
        global_type: SessionType,
        participant: ParticipantId,
    ) -> SessionType:
        """Internal projection implementation."""

        if isinstance(global_type, EndType):
            return LocalEndType()

        elif isinstance(global_type, VariableType):
            return LocalVariableType(name=global_type.name)

        elif isinstance(global_type, MessageType):
            return self._project_message(global_type, participant)

        elif isinstance(global_type, ChoiceType):
            return self._project_choice(global_type, participant)

        elif isinstance(global_type, RecursionType):
            return self._project_recursion(global_type, participant)

        elif isinstance(global_type, ParallelType):
            return self._project_parallel(global_type, participant)

        else:
            raise ProjectionError(
                message=f"Unknown global type: {type(global_type).__name__}",
                participant=participant,
                global_type=global_type,
            )

    def _project_message(
        self,
        msg_type: MessageType,
        participant: ParticipantId,
    ) -> SessionType:
        """Project message type.

        Rules:
        - (p → q : M.G) ↓ p = !q(M).(G ↓ p)  [sender]
        - (p → q : M.G) ↓ q = ?p(M).(G ↓ q)  [receiver]
        - (p → q : M.G) ↓ r = G ↓ r           [other]
        """
        continuation = self._project(msg_type.continuation, participant)

        if participant == msg_type.sender:
            # Sender: !receiver(payload).continuation
            return SendType(
                receiver=msg_type.receiver,
                payload=msg_type.payload,
                continuation=continuation,
            )

        elif participant == msg_type.receiver:
            # Receiver: ?sender(payload).continuation
            return ReceiveType(
                sender=msg_type.sender,
                payload=msg_type.payload,
                continuation=continuation,
            )

        else:
            # Not involved: skip to continuation
            return continuation

    def _project_choice(
        self,
        choice_type: ChoiceType,
        participant: ParticipantId,
    ) -> SessionType:
        """Project choice type.

        Rules:
        - (p → q : {lᵢ: Gᵢ}) ↓ p = ⊕q{lᵢ: Gᵢ ↓ p}  [sender: select]
        - (p → q : {lᵢ: Gᵢ}) ↓ q = &p{lᵢ: Gᵢ ↓ q}  [receiver: branch]
        - (p → q : {lᵢ: Gᵢ}) ↓ r = merge(Gᵢ ↓ r)     [other: merge]
        """
        projected_branches = {
            label: self._project(branch, participant)
            for label, branch in choice_type.branches.items()
        }

        if participant == choice_type.sender:
            # Sender: select type
            return SelectType(receiver=choice_type.receiver, branches=projected_branches)

        elif participant == choice_type.receiver:
            # Receiver: branch type
            return BranchType(sender=choice_type.sender, branches=projected_branches)

        else:
            # Not involved: merge branches
            return self._merge_branches(projected_branches, participant, choice_type)

    def _merge_branches(
        self,
        branches: dict[MessageLabel, SessionType],
        participant: ParticipantId,
        original: ChoiceType,
    ) -> SessionType:
        """Merge branches for a participant not involved in the choice.

        For projection to be defined, all branches must project to
        the same local type for uninvolved participants (merge condition).
        """
        if not branches:
            raise ProjectionError(
                message="Cannot merge empty branches",
                participant=participant,
                global_type=original,
            )

        # All branches must be equivalent for uninvolved participants
        branch_list = list(branches.values())
        first = branch_list[0]

        for i, branch in enumerate(branch_list[1:], 1):
            if not self._types_equivalent(first, branch):
                raise ProjectionError(
                    message=(
                        f"Merge condition violated: branches {list(branches.keys())[0]} "
                        f"and {list(branches.keys())[i]} differ for participant {participant}"
                    ),
                    participant=participant,
                    global_type=original,
                )

        return first

    def _types_equivalent(self, t1: SessionType, t2: SessionType) -> bool:
        """Check if two local types are equivalent (for merge condition)."""
        # Simple structural equality for now
        # Could be extended to handle alpha-equivalence
        return t1 == t2

    def _project_recursion(
        self,
        rec_type: RecursionType,
        participant: ParticipantId,
    ) -> SessionType:
        """Project recursion type.

        Rule: (μX.G) ↓ p = μX.(G ↓ p) if p ∈ participants(G)
                        = end           otherwise
        """
        # Check if participant is involved in the body
        if participant not in rec_type.body.participants():
            return LocalEndType()

        # Prevent infinite recursion
        if rec_type.variable in self._recursion_stack:
            return LocalVariableType(name=rec_type.variable)

        self._recursion_stack.add(rec_type.variable)
        try:
            body = self._project(rec_type.body, participant)
            return LocalRecursionType(variable=rec_type.variable, body=body)
        finally:
            self._recursion_stack.discard(rec_type.variable)

    def _project_parallel(
        self,
        par_type: ParallelType,
        participant: ParticipantId,
    ) -> SessionType:
        """Project parallel composition.

        For parallel types, project both sides and check consistency.
        A participant should only be in one branch.
        """
        in_left = participant in par_type.left.participants()
        in_right = participant in par_type.right.participants()

        if in_left and in_right:
            raise ProjectionError(
                message=f"Participant {participant} appears in both parallel branches",
                participant=participant,
                global_type=par_type,
            )

        if in_left:
            return self._project(par_type.left, participant)
        elif in_right:
            return self._project(par_type.right, participant)
        else:
            return LocalEndType()

    def project_all(
        self,
        global_type: SessionType,
        participants: set[ParticipantId] | list[str] | None = None,
    ) -> dict[ParticipantId, SessionType]:
        """Project global type to local types for all (or specified) participants.

        Args:
            global_type: Global session type
            participants: Optional subset of participants to project.
                If None, auto-discovers all participants from the global type.

        Returns:
            Dictionary mapping each participant to their local type

        Raises:
            ProjectionError: If projection fails for any participant
        """
        if participants is None:
            target_participants = global_type.participants()
        else:
            target_participants = {
                ParticipantId(p) if isinstance(p, str) else p for p in participants
            }
        results: dict[ParticipantId, SessionType] = {}
        for p in target_participants:
            local = self.project(global_type, p)
            if local is None:
                raise ProjectionError(
                    message=f"Projection undefined for participant {p}",
                    participant=p,
                    global_type=global_type,
                )
            results[p] = local
        return results


# Convenience functions


def project(
    global_type: SessionType,
    participant: ParticipantId | str,
    strict: bool = True,
) -> SessionType | None:
    """Project global type to local type for a participant.

    Args:
        global_type: Global session type
        participant: Participant ID
        strict: If True, raise on failure

    Returns:
        Local type or None if undefined
    """
    projector = Projector(strict=strict)
    return projector.project(global_type, ParticipantId(participant))


def project_all(
    global_type: SessionType,
    participants: set[ParticipantId] | list[str] | None = None,
) -> dict[ParticipantId, SessionType]:
    """Project global type to local types for all (or specified) participants.

    Args:
        global_type: Global session type
        participants: Optional subset of participants to project.
            If None, auto-discovers all participants from the global type.

    Returns:
        Dictionary of local types per participant
    """
    projector = Projector(strict=True)
    return projector.project_all(global_type, participants=participants)


__all__ = [
    # Local types
    "LocalEndType",
    "SendType",
    "ReceiveType",
    "SelectType",
    "BranchType",
    "LocalRecursionType",
    "LocalVariableType",
    # Projection
    "Projector",
    "project",
    "project_all",
]
