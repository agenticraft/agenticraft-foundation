"""Scatter-Gather pattern for coordinator-worker communication.

A coordinator sends tasks to multiple workers and gathers their responses.

Global Type (for 2 workers):
    Coordinator → Worker1 : Task.
    Coordinator → Worker2 : Task.
    Worker1 → Coordinator : Result.
    Worker2 → Coordinator : Result.
    end

This pattern is useful for:
- Parallel task distribution
- Map-reduce style computation
- Voting/consensus preparation
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from agenticraft_foundation.mpst.global_types import (
    EndType,
    msg,
)
from agenticraft_foundation.mpst.types import (
    ParticipantId,
    SessionType,
)


@dataclass
class ScatterGatherPattern:
    """Scatter-Gather session pattern.

    Attributes:
        coordinator: Coordinator participant ID
        workers: List of worker participant IDs
        task_label: Label for task messages
        result_label: Label for result messages
    """

    coordinator: ParticipantId | str = "coordinator"
    workers: Sequence[ParticipantId | str] = field(default_factory=lambda: ["worker1", "worker2"])
    task_label: str = "task"
    result_label: str = "result"

    def __post_init__(self) -> None:
        self.coordinator = ParticipantId(self.coordinator)
        self.workers = [ParticipantId(w) for w in self.workers]

    def global_type(self) -> SessionType:
        """Build the global session type.

        Returns:
            Global type for scatter-gather

        The type is:
            Coordinator → Worker_1 : Task.
            ...
            Coordinator → Worker_n : Task.
            Worker_1 → Coordinator : Result.
            ...
            Worker_n → Coordinator : Result.
            end
        """
        if not self.workers:
            return EndType()

        # Build from end backwards
        # First: gather phase (workers send results)
        current: SessionType = EndType()
        for worker in reversed(self.workers):
            current = msg(worker, self.coordinator, self.result_label, current)

        # Then: scatter phase (coordinator sends tasks)
        for worker in reversed(self.workers):
            current = msg(self.coordinator, worker, self.task_label, current)

        return current

    def participants(self) -> set[ParticipantId]:
        """Get all participants in the scatter-gather pattern.

        Returns:
            Set containing the coordinator and all worker identifiers.
        """
        return {ParticipantId(self.coordinator), *(ParticipantId(w) for w in self.workers)}


def scatter_gather(
    coordinator: str = "coordinator",
    workers: Sequence[str] | None = None,
    task_label: str = "task",
    result_label: str = "result",
) -> SessionType:
    """Create a scatter-gather global type.

    Args:
        coordinator: Coordinator participant name
        workers: List of worker names (default: ["worker1", "worker2"])
        task_label: Label for task messages
        result_label: Label for result messages

    Returns:
        Global session type

    Example:
        # Simple 2-worker scatter-gather
        g = scatter_gather()

        # Custom workers
        g = scatter_gather("master", ["node1", "node2", "node3"])
    """
    if workers is None:
        workers = ["worker1", "worker2"]

    pattern = ScatterGatherPattern(
        coordinator=coordinator,
        workers=workers,
        task_label=task_label,
        result_label=result_label,
    )
    return pattern.global_type()


__all__ = [
    "ScatterGatherPattern",
    "scatter_gather",
]
