"""Consensus pattern for multi-party agreement protocols.

Two-Phase Commit (2PC) protocol where a coordinator collects
votes from participants and then broadcasts a decision.

Global Type (for 2 participants):
    Coordinator → Participant1 : Prepare.
    Coordinator → Participant2 : Prepare.
    Participant1 → Coordinator : {
        vote_yes: Participant2 → Coordinator : {
            vote_yes: Coordinator → Participant1 : Commit.
                      Coordinator → Participant2 : Commit.
                      end,
            vote_no:  Coordinator → Participant1 : Abort.
                      Coordinator → Participant2 : Abort.
                      end
        },
        vote_no: Participant2 → Coordinator : {
            vote_yes: Coordinator → Participant1 : Abort.
                      Coordinator → Participant2 : Abort.
                      end,
            vote_no:  Coordinator → Participant1 : Abort.
                      Coordinator → Participant2 : Abort.
                      end
        }
    }

Simplified version used here:
    Phase 1: Coordinator → All : Prepare
    Phase 2: All → Coordinator : Vote
    Phase 3: Coordinator → All : Decision
    end

This pattern is useful for:
- Distributed transactions
- Multi-party agreement
- Atomic commitment protocols
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from agenticraft_foundation.mpst.global_types import (
    EndType,
    MessageType,
    choice,
    msg,
)
from agenticraft_foundation.mpst.types import (
    ParticipantId,
    SessionType,
)


@dataclass
class ConsensusPattern:
    """Two-Phase Commit consensus pattern.

    Attributes:
        coordinator: Coordinator participant ID
        participants: List of participant IDs
        prepare_label: Label for prepare messages
        vote_yes_label: Label for affirmative vote
        vote_no_label: Label for negative vote
        commit_label: Label for commit decision
        abort_label: Label for abort decision
    """

    coordinator: ParticipantId | str = "coordinator"
    participants: Sequence[ParticipantId | str] = field(
        default_factory=lambda: ["participant1", "participant2"]
    )
    prepare_label: str = "prepare"
    vote_yes_label: str = "vote_yes"
    vote_no_label: str = "vote_no"
    commit_label: str = "commit"
    abort_label: str = "abort"

    _coordinator: ParticipantId | None = field(default=None, init=False, repr=False)
    _participants: list[ParticipantId] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._coordinator = ParticipantId(self.coordinator)
        self._participants = [ParticipantId(p) for p in self.participants]
        if len(self._participants) < 1:
            raise ValueError("Consensus requires at least 1 participant")

    def _build_decision_phase(self, decision_label: str) -> SessionType:
        """Build the decision broadcast phase."""
        coord = self._coordinator
        parts = self._participants or []
        if coord is None:
            raise ValueError("Pattern not initialized")
        current: SessionType = EndType()
        for participant in reversed(parts):
            current = msg(coord, participant, decision_label, current)
        return current

    def _build_simple_voting(self) -> SessionType:
        """Build simplified voting phase (sequential votes, then decision).

        This is a simplified 2PC where:
        1. All participants vote sequentially
        2. Coordinator makes decision based on votes
        3. Decision is broadcast to all

        A real 2PC would have branching for each vote combination,
        but that leads to exponential explosion with many participants.
        """
        coord = self._coordinator
        parts = self._participants or []
        if coord is None:
            raise ValueError("Pattern not initialized")

        # Phase 3: Coordinator broadcasts decision
        # For simplicity, we model just the commit path
        # (In practice, runtime handles abort)
        commit_phase = self._build_decision_phase(self.commit_label)

        # Phase 2: All participants vote (sequentially for simplicity)
        voting_phase = commit_phase
        for participant in reversed(parts):
            voting_phase = msg(participant, coord, self.vote_yes_label, voting_phase)

        # Phase 1: Coordinator sends prepare to all
        prepare_phase = voting_phase
        for participant in reversed(parts):
            prepare_phase = msg(coord, participant, self.prepare_label, prepare_phase)

        return prepare_phase

    def _build_with_choice(self) -> SessionType:
        """Build 2PC with explicit choice branches.

        Only practical for 1-2 participants due to exponential branches.
        """
        coord = self._coordinator
        parts = self._participants or []
        if coord is None:
            raise ValueError("Pattern not initialized")

        if len(parts) > 2:
            # Fall back to simple voting for many participants
            return self._build_simple_voting()

        # Build the decision phases
        commit_phase = self._build_decision_phase(self.commit_label)
        abort_phase = self._build_decision_phase(self.abort_label)

        if len(parts) == 1:
            # Single participant: simple choice
            participant = parts[0]
            voting = choice(
                participant,
                coord,
                {
                    self.vote_yes_label: commit_phase,
                    self.vote_no_label: abort_phase,
                },
            )
            prepare = msg(coord, participant, self.prepare_label, voting)
            return prepare

        # Two participants: nested choices
        p1, p2 = parts[0], parts[1]

        # P2's vote after P1 voted yes
        p2_after_yes = choice(
            p2,
            coord,
            {
                self.vote_yes_label: commit_phase,  # Both yes -> commit
                self.vote_no_label: abort_phase,  # P1 yes, P2 no -> abort
            },
        )

        # P2's vote after P1 voted no
        p2_after_no = choice(
            p2,
            coord,
            {
                self.vote_yes_label: abort_phase,  # P1 no, P2 yes -> abort
                self.vote_no_label: abort_phase,  # Both no -> abort
            },
        )

        # P1's vote with nested P2 choices
        p1_vote = choice(
            p1,
            coord,
            {
                self.vote_yes_label: p2_after_yes,
                self.vote_no_label: p2_after_no,
            },
        )

        # Prepare phase
        prepare = msg(coord, p1, self.prepare_label)
        prepare = MessageType(
            sender=coord,
            receiver=p1,
            payload=prepare.payload,
            continuation=msg(coord, p2, self.prepare_label, p1_vote),
        )

        return prepare

    def global_type(self, with_choice: bool = False) -> SessionType:
        """Build the global session type.

        Args:
            with_choice: If True, include explicit choice branches
                        (only practical for 1-2 participants)

        Returns:
            Global type for two-phase commit
        """
        if with_choice:
            return self._build_with_choice()
        return self._build_simple_voting()

    def participants_set(self) -> set[ParticipantId]:
        """Get all participants including coordinator."""
        coord = self._coordinator
        parts = self._participants or []
        if coord is None:
            return set(parts)
        return {coord} | set(parts)


def two_phase_commit(
    coordinator: str = "coordinator",
    participants: Sequence[str] | None = None,
    with_choice: bool = False,
) -> SessionType:
    """Create a two-phase commit global type.

    Args:
        coordinator: Coordinator participant name
        participants: List of participant names
        with_choice: Include explicit choice branches (only for 1-2 participants)

    Returns:
        Global session type

    Example:
        # Simple 2PC with 2 participants
        g = two_phase_commit()

        # Custom participants
        g = two_phase_commit("leader", ["replica1", "replica2", "replica3"])

        # With explicit branching (for 2 participants)
        g = two_phase_commit(with_choice=True)
    """
    if participants is None:
        participants = ["participant1", "participant2"]

    pattern = ConsensusPattern(
        coordinator=coordinator,
        participants=participants,
    )
    return pattern.global_type(with_choice=with_choice)


__all__ = [
    "ConsensusPattern",
    "two_phase_commit",
]
