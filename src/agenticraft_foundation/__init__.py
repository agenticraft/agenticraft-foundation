"""
agenticraft-foundation -- Formally verified mathematical foundations
for multi-agent AI coordination.

13 CSP operators | Multiparty Session Types | Spectral Topology | Protocol Analysis

Minimal dependencies (NumPy). Pure Python. 1250+ tests.
"""

from agenticraft_foundation._version import __version__
from agenticraft_foundation.algebra import (
    TIMEOUT_EVENT,
    Event,
    ExternalChoice,
    Guard,
    Hiding,
    InternalChoice,
    Interrupt,
    Parallel,
    Pipe,
    Prefix,
    Process,
    ProcessKind,
    Recursion,
    Rename,
    Sequential,
    Skip,
    Stop,
    Timeout,
    Variable,
    build_lts,
    detect_deadlock,
    is_deadlock_free,
    substitute,
    trace_equivalent,
    traces,
)
from agenticraft_foundation.mpst import SessionMonitor
from agenticraft_foundation.topology import HypergraphNetwork, LaplacianAnalysis, NetworkGraph
from agenticraft_foundation.verification import InvariantRegistry

# NOTE: Full subpackage APIs are accessible via direct imports:
#   from agenticraft_foundation.algebra import trace_refines, failures_refines, ...
#   from agenticraft_foundation.topology import compare_topologies, ...
#   from agenticraft_foundation.mpst import Projector, SessionTypeChecker, ...
#   from agenticraft_foundation.protocols import ...
#   from agenticraft_foundation.verification import InvariantChecker, ...
#   from agenticraft_foundation.complexity import ...
#   from agenticraft_foundation.specifications import ...

__all__ = [
    "__version__",
    # CSP Core types
    "Event",
    "ProcessKind",
    "Process",
    # CSP Primitives (8)
    "Stop",
    "Skip",
    "Prefix",
    "ExternalChoice",
    "InternalChoice",
    "Parallel",
    "Sequential",
    "Hiding",
    "Recursion",
    "Variable",
    "substitute",
    # Agent-Specific Extensions (5)
    "Interrupt",
    "Timeout",
    "Guard",
    "Rename",
    "Pipe",
    "TIMEOUT_EVENT",
    # Semantics & Analysis
    "traces",
    "trace_equivalent",
    "build_lts",
    "detect_deadlock",
    "is_deadlock_free",
    # Topology
    "LaplacianAnalysis",
    "NetworkGraph",
    "HypergraphNetwork",
    # Session Types
    "SessionMonitor",
    # Verification
    "InvariantRegistry",
]
