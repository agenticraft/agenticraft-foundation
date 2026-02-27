"""Verification tools for distributed systems.

This module provides:
- Runtime invariant checking and state transition monitoring
- Structured counterexample generation for refinement/equivalence failures
- CTL temporal logic model checking
- Probabilistic verification (DTMC reachability, steady-state, expected steps)
"""

from __future__ import annotations

from agenticraft_foundation.verification.counterexamples import (
    AnnotatedStep,
    CounterexampleExplanation,
    StepStatus,
    explain_equivalence_failure,
    explain_refinement_failure,
    find_minimal_counterexample,
)
from agenticraft_foundation.verification.invariant_checker import (
    Invariant,
    InvariantRegistry,
    StateTransitionMonitor,
    Violation,
    ViolationSeverity,
    assert_invariant,
    check_invariant,
    invariant,
    register_invariant,
)
from agenticraft_foundation.verification.probabilistic import (
    DTMC,
    DTMCState,
    ExpectedStepsResult,
    ProbabilisticResult,
    ProbabilisticTransition,
    SteadyStateResult,
    build_dtmc_from_lts,
    check_reachability,
    expected_steps,
    steady_state,
)
from agenticraft_foundation.verification.temporal import (
    AF,
    AG,
    AU,
    AX,
    EF,
    EG,
    EU,
    EX,
    And,
    Atomic,
    CTLFormula,
    Implies,
    Labeling,
    ModelCheckResult,
    Not,
    Or,
    check_liveness,
    check_safety,
    model_check,
)
from agenticraft_foundation.verification.temporal import (
    check_invariant as check_temporal_invariant,  # noqa: F811
)

__all__ = [
    # --- Invariant Checker ---
    "ViolationSeverity",
    "Invariant",
    "Violation",
    "InvariantRegistry",
    "register_invariant",
    "check_invariant",
    "invariant",
    "assert_invariant",
    "StateTransitionMonitor",
    # --- Counterexamples ---
    "StepStatus",
    "AnnotatedStep",
    "CounterexampleExplanation",
    "explain_refinement_failure",
    "explain_equivalence_failure",
    "find_minimal_counterexample",
    # --- Temporal Logic (CTL) ---
    "CTLFormula",
    "Atomic",
    "Not",
    "And",
    "Or",
    "Implies",
    "EX",
    "EF",
    "EG",
    "EU",
    "AX",
    "AF",
    "AG",
    "AU",
    "Labeling",
    "ModelCheckResult",
    "model_check",
    "check_safety",
    "check_liveness",
    "check_temporal_invariant",
    # --- Probabilistic (DTMC) ---
    "ProbabilisticTransition",
    "DTMCState",
    "DTMC",
    "ProbabilisticResult",
    "SteadyStateResult",
    "ExpectedStepsResult",
    "check_reachability",
    "steady_state",
    "expected_steps",
    "build_dtmc_from_lts",
]
