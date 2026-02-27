"""Pipeline pattern for sequential multi-stage processing.

A message flows through a sequence of processing stages,
with each stage passing data to the next.

Global Type (for 3 stages):
    Stage1 → Stage2 : Data.
    Stage2 → Stage3 : Data.
    end

This pattern is useful for:
- Data transformation pipelines
- Sequential processing chains
- Assembly line workflows
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from agenticraft_foundation.mpst.global_types import (
    EndType,
    msg,
    rec,
    var,
)
from agenticraft_foundation.mpst.types import (
    ParticipantId,
    SessionType,
)


@dataclass
class PipelinePattern:
    """Pipeline session pattern.

    Attributes:
        stages: List of stage participant IDs in order
        data_label: Label for data messages between stages
        repeatable: If True, pipeline can repeat
    """

    stages: Sequence[ParticipantId | str] = field(
        default_factory=lambda: ["stage1", "stage2", "stage3"]
    )
    data_label: str = "data"
    repeatable: bool = False

    _stages: list[ParticipantId] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._stages = [ParticipantId(s) for s in self.stages]
        if len(self._stages) < 2:
            raise ValueError("Pipeline requires at least 2 stages")

    def global_type(self) -> SessionType:
        """Build the global session type.

        Returns:
            Global type for pipeline

        The type is:
            Stage_1 → Stage_2 : Data.
            Stage_2 → Stage_3 : Data.
            ...
            Stage_{n-1} → Stage_n : Data.
            end (or X for repeatable)
        """
        stages = self._stages or []
        if len(stages) < 2:
            return EndType()

        # Build the message chain
        if self.repeatable:
            # μX. Stage1 → Stage2 : Data. ... Stage_n-1 → Stage_n : Data. X
            continuation: SessionType = var("X")
        else:
            continuation = EndType()

        # Build from end backwards
        for i in range(len(stages) - 2, -1, -1):
            sender = stages[i]
            receiver = stages[i + 1]
            continuation = msg(sender, receiver, self.data_label, continuation)

        if self.repeatable:
            return rec("X", continuation)

        return continuation

    def participants(self) -> set[ParticipantId]:
        """Get all participants in the pipeline.

        Returns:
            Set of stage participant identifiers.
        """
        return set(self._stages or [])


def pipeline(
    stages: Sequence[str] | None = None,
    data_label: str = "data",
    repeatable: bool = False,
) -> SessionType:
    """Create a pipeline global type.

    Args:
        stages: List of stage names (default: ["stage1", "stage2", "stage3"])
        data_label: Label for data messages
        repeatable: If True, pipeline can repeat

    Returns:
        Global session type

    Example:
        # Simple 3-stage pipeline
        g = pipeline()

        # Custom stages
        g = pipeline(["fetch", "process", "store", "notify"])

        # Continuous processing loop
        g = pipeline(["input", "transform", "output"], repeatable=True)
    """
    if stages is None:
        stages = ["stage1", "stage2", "stage3"]

    pattern = PipelinePattern(
        stages=stages,
        data_label=data_label,
        repeatable=repeatable,
    )
    return pattern.global_type()


__all__ = [
    "PipelinePattern",
    "pipeline",
]
