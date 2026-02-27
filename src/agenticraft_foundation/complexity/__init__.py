"""Complexity annotations and bounds for distributed algorithms.

This module provides tools for documenting and analyzing algorithm complexity,
including decorators for annotation and utilities for complexity comparison.
"""

from __future__ import annotations

from agenticraft_foundation.complexity.annotations import (
    ComplexityAnalysis,
    ComplexityBound,
    ComplexityClass,
    DistributedComplexity,
    FaultModel,
    SynchronyModel,
    analyze_complexity,
    compare_complexity,
    complexity,
    consensus_complexity,
    get_all_complexities,
    get_complexity,
    gossip_complexity,
    parse_big_o,
)
from agenticraft_foundation.complexity.bounds import (
    BROADCAST_BOUNDS,
    CONSENSUS_BOUNDS,
    GOSSIP_BOUNDS,
    IMPOSSIBILITY_RESULTS,
    LEADER_ELECTION_BOUNDS,
    BoundType,
    ComplexityComparison,
    ImpossibilityResult,
    OptimalityCheck,
    TheoreticalBound,
    check_optimality,
    compare_algorithms,
    get_impossibility,
    validate_fault_tolerance,
)

__all__ = [
    # Complexity annotations
    "ComplexityClass",
    "SynchronyModel",
    "FaultModel",
    "ComplexityBound",
    "DistributedComplexity",
    "complexity",
    "consensus_complexity",
    "gossip_complexity",
    "get_complexity",
    "get_all_complexities",
    "ComplexityAnalysis",
    "analyze_complexity",
    "parse_big_o",
    "compare_complexity",
    # Complexity bounds
    "BoundType",
    "TheoreticalBound",
    "CONSENSUS_BOUNDS",
    "GOSSIP_BOUNDS",
    "LEADER_ELECTION_BOUNDS",
    "BROADCAST_BOUNDS",
    "OptimalityCheck",
    "check_optimality",
    "ImpossibilityResult",
    "IMPOSSIBILITY_RESULTS",
    "get_impossibility",
    "validate_fault_tolerance",
    "ComplexityComparison",
    "compare_algorithms",
]
