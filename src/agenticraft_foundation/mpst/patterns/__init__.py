"""Common multi-agent session patterns.

This module provides reusable session type patterns for common
multi-agent communication scenarios.

Patterns:
- RequestResponse: Simple client-server request-response
- ScatterGather: Coordinator sends to all, gathers responses
- Pipeline: Sequential processing through stages
- Consensus: Multi-party agreement protocol

Each pattern provides:
- Global type constructor
- Local type projections
- Example usage
"""

from agenticraft_foundation.mpst.patterns.consensus import (
    ConsensusPattern,
    two_phase_commit,
)
from agenticraft_foundation.mpst.patterns.pipeline import (
    PipelinePattern,
    pipeline,
)
from agenticraft_foundation.mpst.patterns.request_response import (
    RequestResponsePattern,
    request_response,
)
from agenticraft_foundation.mpst.patterns.scatter_gather import (
    ScatterGatherPattern,
    scatter_gather,
)

__all__ = [
    # Request-Response
    "RequestResponsePattern",
    "request_response",
    # Scatter-Gather
    "ScatterGatherPattern",
    "scatter_gather",
    # Pipeline
    "PipelinePattern",
    "pipeline",
    # Consensus
    "ConsensusPattern",
    "two_phase_commit",
]
