"""Process Algebra module for formal verification of agent coordination.

This module provides:
- CSP (Communicating Sequential Processes) primitives
- Labeled Transition Systems (LTS) for operational semantics
- Process equivalence (trace equivalence, bisimulation)
- Refinement checking for specification verification
- Coordination patterns for multi-agent systems

Based on:
- Hoare (1985) - Communicating Sequential Processes
- Milner (1980) - Calculus of Communicating Systems
- Roscoe (1998) - The Theory and Practice of Concurrency
"""

from __future__ import annotations

from .csp import (
    TAU,
    TICK,
    Event,
    ExternalChoice,
    Hiding,
    InternalChoice,
    Parallel,
    Prefix,
    Process,
    ProcessKind,
    Recursion,
    Sequential,
    Skip,
    Stop,
    Variable,
    choice,
    hide,
    interleave,
    internal_choice,
    parallel,
    prefix,
    rec,
    sequential,
    skip,
    stop,
    substitute,
    var,
)
from .equivalence import (
    EquivalenceResult,
    Failure,
    are_equivalent,
    failures,
    failures_equivalent,
    strong_bisimilar,
    trace_equivalent,
    weak_bisimilar,
)
from .operators import (
    TIMEOUT_EVENT,
    Guard,
    Interrupt,
    Pipe,
    Rename,
    Timeout,
)
from .patterns import (
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
from .refinement import (
    DivergenceResult,
    RefinementReport,
    RefinementResult,
    analyze_refinement,
    check_divergence_free,
    failures_refines,
    fd_refines,
    refines,
    trace_refines,
)
from .semantics import (
    LTS,
    DeadlockAnalysis,
    LivenessAnalysis,
    LTSBuilder,
    LTSState,
    Trace,
    Transition,
    accepts,
    analyze_liveness,
    build_lts,
    detect_deadlock,
    is_deadlock_free,
    maximal_traces,
    traces,
)

__all__ = [
    # CSP Core types
    "Event",
    "TICK",
    "TAU",
    "ProcessKind",
    "Process",
    # CSP Primitives
    "Stop",
    "Skip",
    "Prefix",
    # CSP Choice
    "ExternalChoice",
    "InternalChoice",
    # CSP Parallel
    "Parallel",
    # CSP Sequential
    "Sequential",
    # CSP Hiding
    "Hiding",
    # CSP Recursion
    "Recursion",
    "Variable",
    "substitute",
    # CSP Constructors
    "stop",
    "skip",
    "prefix",
    "choice",
    "internal_choice",
    "parallel",
    "interleave",
    "sequential",
    "hide",
    "rec",
    "var",
    # LTS / Semantics
    "Transition",
    "LTSState",
    "LTS",
    "LTSBuilder",
    "Trace",
    "traces",
    "maximal_traces",
    "accepts",
    "DeadlockAnalysis",
    "detect_deadlock",
    "LivenessAnalysis",
    "analyze_liveness",
    "build_lts",
    "is_deadlock_free",
    # Equivalence
    "EquivalenceResult",
    "Failure",
    "trace_equivalent",
    "strong_bisimilar",
    "weak_bisimilar",
    "failures",
    "failures_equivalent",
    "are_equivalent",
    # Refinement
    "RefinementResult",
    "RefinementReport",
    "DivergenceResult",
    "trace_refines",
    "failures_refines",
    "fd_refines",
    "check_divergence_free",
    "analyze_refinement",
    "refines",
    # Patterns
    "RequestResponsePattern",
    "request_response",
    "PipelinePattern",
    "pipeline",
    "ScatterGatherPattern",
    "scatter_gather",
    "BarrierPattern",
    "barrier",
    "MutexPattern",
    "mutex",
    "ProducerConsumerPattern",
    "producer_consumer",
    "compose_agents",
    "verify_pattern",
    # Agent-Specific Extensions
    "Interrupt",
    "Timeout",
    "Guard",
    "Rename",
    "Pipe",
    "TIMEOUT_EVENT",
]
