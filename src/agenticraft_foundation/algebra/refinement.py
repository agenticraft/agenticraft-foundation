"""Refinement checking for CSP processes.

This module provides:
- Trace refinement: SPEC ⊑_T IMPL iff traces(IMPL) ⊆ traces(SPEC)
- Failures refinement: SPEC ⊑_F IMPL iff failures(IMPL) ⊆ failures(SPEC)
- Failures-divergences refinement (FD): Standard CSP refinement

Refinement is the key relation for verification:
- SPEC describes allowed behaviors
- IMPL must not have behaviors outside SPEC
- If IMPL ⊑ SPEC, IMPL is a valid implementation

Note on direction:
- In CSP: SPEC ⊑_T IMPL reads "SPEC is trace-refined by IMPL"
- This means traces(IMPL) ⊆ traces(SPEC): IMPL has fewer behaviors
- Equivalently: IMPL does nothing outside what SPEC allows
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .csp import TAU, Event, Process
from .equivalence import Failure, failures
from .semantics import LTS, LTSBuilder, Trace, accepts, traces

# =============================================================================
# Refinement Results
# =============================================================================


@dataclass
class RefinementResult:
    """Result of a refinement check."""

    refines: bool
    counterexample: Trace | Failure | None = None
    message: str = ""

    @property
    def is_valid(self) -> bool:
        """True if implementation refines specification."""
        return self.refines


@dataclass
class RefinementReport:
    """Detailed refinement analysis report."""

    spec_name: str
    impl_name: str
    trace_refinement: RefinementResult
    failures_refinement: RefinementResult | None = None
    spec_traces_sample: list[Trace] = field(default_factory=list)
    impl_traces_sample: list[Trace] = field(default_factory=list)
    impl_extra_traces: list[Trace] = field(default_factory=list)

    @property
    def overall_result(self) -> bool:
        """True if all refinement checks pass."""
        if self.failures_refinement:
            return self.trace_refinement.is_valid and self.failures_refinement.is_valid
        return self.trace_refinement.is_valid


# =============================================================================
# Trace Refinement
# =============================================================================


def trace_refines(
    spec: Process | LTS,
    impl: Process | LTS,
    max_trace_length: int = 100,
) -> RefinementResult:
    """Check trace refinement: traces(IMPL) ⊆ traces(SPEC).

    An implementation trace-refines a specification if every trace
    of the implementation is also a trace of the specification.

    Args:
        spec: The specification process
        impl: The implementation process
        max_trace_length: Maximum trace length to check

    Returns:
        RefinementResult with counterexample if refinement fails
    """
    lts_spec = _to_lts(spec)
    lts_impl = _to_lts(impl)

    # Get all implementation traces
    impl_traces = set(traces(lts_impl, max_length=max_trace_length))

    # Check each implementation trace is accepted by spec
    for trace in impl_traces:
        if not accepts(lts_spec, trace):
            return RefinementResult(
                refines=False,
                counterexample=trace,
                message=f"Implementation trace {_trace_str(trace)} not in specification",
            )

    return RefinementResult(
        refines=True,
        message="All implementation traces are valid specification traces",
    )


# =============================================================================
# Failures Refinement
# =============================================================================


def failures_refines(
    spec: Process | LTS,
    impl: Process | LTS,
    max_trace_length: int = 50,
) -> RefinementResult:
    """Check failures refinement: failures(IMPL) ⊆ failures(SPEC).

    An implementation failures-refines a specification if every failure
    of the implementation is also a failure of the specification.

    This is stronger than trace refinement: not only must traces match,
    but refusal sets must also be subsets.

    Args:
        spec: The specification process
        impl: The implementation process
        max_trace_length: Maximum trace length

    Returns:
        RefinementResult with counterexample if refinement fails
    """
    lts_spec = _to_lts(spec)
    lts_impl = _to_lts(impl)

    # Get specification failures
    spec_failures = set(failures(lts_spec, max_trace_length))

    # Check each implementation failure
    for impl_failure in failures(lts_impl, max_trace_length):
        # First, the trace must be in spec
        if not accepts(lts_spec, impl_failure.trace):
            return RefinementResult(
                refines=False,
                counterexample=impl_failure,
                message=f"Implementation trace {_trace_str(impl_failure.trace)} not in spec",
            )

        # Then, check if this failure is allowed by spec
        # (impl can refuse only if spec also refuses)
        if impl_failure not in spec_failures:
            # Check if refusals are subset of any matching spec failure
            matching = [f for f in spec_failures if f.trace == impl_failure.trace]
            if not matching:
                return RefinementResult(
                    refines=False,
                    counterexample=impl_failure,
                    message=f"No matching spec failure for trace {_trace_str(impl_failure.trace)}",
                )

            # Check if refusals are valid
            spec_refusals: set[Event] = set()
            for f in matching:
                spec_refusals.update(f.refusals)

            extra_refusals = impl_failure.refusals - spec_refusals
            if extra_refusals:
                return RefinementResult(
                    refines=False,
                    counterexample=impl_failure,
                    message=(
                        f"Implementation refuses {extra_refusals}"
                        f" after {_trace_str(impl_failure.trace)}"
                    ),
                )

    return RefinementResult(
        refines=True,
        message="Implementation failures are subset of specification failures",
    )


# =============================================================================
# Divergence-Freedom
# =============================================================================


@dataclass
class DivergenceResult:
    """Result of divergence check."""

    is_divergence_free: bool
    divergence_trace: Trace | None = None
    divergent_states: list[int] = field(default_factory=list)


def check_divergence_free(
    process: Process | LTS,
    max_depth: int = 1000,
) -> DivergenceResult:
    """Check if process is divergence-free.

    A process diverges if it can perform an infinite sequence of τ actions.
    This is detected by finding τ-cycles in the LTS.

    Args:
        process: The process to check
        max_depth: Maximum exploration depth

    Returns:
        DivergenceResult indicating divergence-freedom
    """
    lts = _to_lts(process)

    # Find τ-cycles using DFS
    divergent_states: list[int] = []

    for state_id in lts.states:
        if _has_tau_cycle(lts, state_id):
            divergent_states.append(state_id)

    if not divergent_states:
        return DivergenceResult(is_divergence_free=True)

    # Find trace to first divergent state
    trace = _find_trace_to_state(lts, divergent_states[0])

    return DivergenceResult(
        is_divergence_free=False,
        divergence_trace=trace,
        divergent_states=divergent_states,
    )


def _has_tau_cycle(lts: LTS, start_state: int) -> bool:
    """Check if there's a τ-cycle from this state."""
    visited: set[int] = set()
    path: set[int] = set()
    stack = [(start_state, False)]

    while stack:
        state, processed = stack.pop()

        if processed:
            path.remove(state)
            continue

        if state in path:
            # Found cycle
            return True

        if state in visited:
            continue

        visited.add(state)
        path.add(state)
        stack.append((state, True))

        # Follow τ transitions
        for event, target in lts.successors(state):
            if event == TAU:
                stack.append((target, False))

    return False


def _find_trace_to_state(lts: LTS, target: int) -> Trace | None:
    """Find a trace from initial state to target."""
    queue: deque[tuple[int, Trace]] = deque()
    queue.append((lts.initial_state, ()))
    seen: set[int] = set()

    while queue:
        state_id, trace = queue.popleft()

        if state_id in seen:
            continue
        seen.add(state_id)

        if state_id == target:
            return trace

        for event, next_state in lts.successors(state_id):
            if event == TAU:
                queue.append((next_state, trace))
            else:
                queue.append((next_state, trace + (event,)))

    return None


# =============================================================================
# FD (Failures-Divergences) Refinement
# =============================================================================


def fd_refines(
    spec: Process | LTS,
    impl: Process | LTS,
    max_trace_length: int = 50,
) -> RefinementResult:
    """Check failures-divergences refinement.

    This is the standard CSP refinement:
    IMPL ⊑_FD SPEC iff:
    1. failures(IMPL) ⊆ failures(SPEC)
    2. divergences(IMPL) ⊆ divergences(SPEC)

    If SPEC is divergence-free, then IMPL must also be divergence-free.

    Args:
        spec: The specification process
        impl: The implementation process
        max_trace_length: Maximum trace length

    Returns:
        RefinementResult
    """
    # First check divergence
    spec_div = check_divergence_free(spec)
    impl_div = check_divergence_free(impl)

    if spec_div.is_divergence_free and not impl_div.is_divergence_free:
        return RefinementResult(
            refines=False,
            counterexample=impl_div.divergence_trace,
            message="Implementation diverges but specification is divergence-free",
        )

    # Then check failures refinement
    return failures_refines(spec, impl, max_trace_length)


# =============================================================================
# Comprehensive Analysis
# =============================================================================


def analyze_refinement(
    spec: Process,
    impl: Process,
    spec_name: str = "SPEC",
    impl_name: str = "IMPL",
    max_trace_length: int = 50,
    check_failures: bool = True,
) -> RefinementReport:
    """Perform comprehensive refinement analysis.

    Args:
        spec: The specification process
        impl: The implementation process
        spec_name: Name for the spec in reports
        impl_name: Name for the impl in reports
        max_trace_length: Maximum trace length
        check_failures: Whether to check failures refinement

    Returns:
        RefinementReport with detailed analysis
    """
    lts_spec = _to_lts(spec)
    lts_impl = _to_lts(impl)

    # Get trace samples
    spec_traces_list = list(traces(lts_spec, max_length=min(max_trace_length, 20)))[:10]
    impl_traces_list = list(traces(lts_impl, max_length=min(max_trace_length, 20)))[:10]

    # Check trace refinement
    trace_result = trace_refines(spec, impl, max_trace_length)

    # Find extra implementation traces
    spec_traces_set = set(traces(lts_spec, max_length=max_trace_length))
    impl_extra = [
        t for t in traces(lts_impl, max_length=max_trace_length) if t not in spec_traces_set
    ][:5]

    # Check failures refinement if requested
    failures_result = None
    if check_failures:
        failures_result = failures_refines(spec, impl, max_trace_length)

    return RefinementReport(
        spec_name=spec_name,
        impl_name=impl_name,
        trace_refinement=trace_result,
        failures_refinement=failures_result,
        spec_traces_sample=spec_traces_list,
        impl_traces_sample=impl_traces_list,
        impl_extra_traces=impl_extra,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _to_lts(p: Process | LTS) -> LTS:
    """Convert process to LTS if needed."""
    if isinstance(p, LTS):
        return p
    return LTSBuilder().build(p)


def _trace_str(trace: Trace) -> str:
    """Format trace for display."""
    if not trace:
        return "⟨⟩"
    return f"⟨{', '.join(str(e) for e in trace)}⟩"


# =============================================================================
# Convenience Functions
# =============================================================================


def refines(
    spec: Process,
    impl: Process,
    mode: str = "trace",
) -> bool:
    """Check if implementation refines specification.

    Args:
        spec: The specification process
        impl: The implementation process
        mode: Refinement type:
            - "trace": Trace refinement
            - "failures": Failures refinement
            - "fd": Failures-divergences refinement

    Returns:
        True if impl refines spec
    """
    checkers = {
        "trace": trace_refines,
        "failures": failures_refines,
        "fd": fd_refines,
    }

    checker = checkers.get(mode)
    if checker is None:
        raise ValueError(f"Unknown refinement mode: {mode}")

    return checker(spec, impl).is_valid


__all__ = [
    # Result types
    "RefinementResult",
    "RefinementReport",
    "DivergenceResult",
    # Refinement checks
    "trace_refines",
    "failures_refines",
    "fd_refines",
    # Divergence
    "check_divergence_free",
    # Analysis
    "analyze_refinement",
    # Convenience
    "refines",
]
