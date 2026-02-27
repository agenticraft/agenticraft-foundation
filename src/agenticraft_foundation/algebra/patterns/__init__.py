"""Coordination patterns for multi-agent systems in CSP.

This module provides:
- Request-Response: Client-server interaction
- Pipeline: Sequential processing stages
- Scatter-Gather: Parallel task distribution and collection
- Barrier: Synchronization point for multiple agents
- Mutex: Mutual exclusion pattern
- Producer-Consumer: Buffered communication
"""

from __future__ import annotations

from .coordination import (
    BarrierPattern,
    MutexPattern,
    PipelinePattern,
    ProducerConsumerPattern,
    RequestResponsePattern,
    ScatterGatherPattern,
    barrier,
    compose_agents,
    mutex,
    pipeline,
    producer_consumer,
    request_response,
    scatter_gather,
    verify_pattern,
)

__all__ = [
    # Request-Response
    "RequestResponsePattern",
    "request_response",
    # Pipeline
    "PipelinePattern",
    "pipeline",
    # Scatter-Gather
    "ScatterGatherPattern",
    "scatter_gather",
    # Barrier
    "BarrierPattern",
    "barrier",
    # Mutex
    "MutexPattern",
    "mutex",
    # Producer-Consumer
    "ProducerConsumerPattern",
    "producer_consumer",
    # Utilities
    "compose_agents",
    "verify_pattern",
]
