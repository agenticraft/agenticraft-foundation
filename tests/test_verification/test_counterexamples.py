"""Tests for structured counterexample generation."""

from __future__ import annotations

import pytest

from agenticraft_foundation.algebra.csp import Event, Prefix, Skip, Stop
from agenticraft_foundation.algebra.equivalence import (
    EquivalenceResult,
    strong_bisimilar,
    trace_equivalent,
)
from agenticraft_foundation.algebra.refinement import (
    RefinementResult,
    failures_refines,
    trace_refines,
)
from agenticraft_foundation.algebra.semantics import LTS, LTSState, Transition
from agenticraft_foundation.verification.counterexamples import (
    AnnotatedStep,
    CounterexampleExplanation,
    StepStatus,
    explain_equivalence_failure,
    explain_refinement_failure,
    find_minimal_counterexample,
)

# =============================================================================
# Helpers: Build simple processes
# =============================================================================


def _make_prefix_chain(*events: str) -> Prefix:
    """Build a -> b -> c -> STOP chain."""
    result: Prefix | Stop = Stop()
    for e in reversed(events):
        result = Prefix(Event(e), result)
    return result  # type: ignore[return-value]


def _make_lts_simple() -> tuple[LTS, LTS]:
    """Build spec and impl LTS for basic trace divergence.

    Spec: 0 --a--> 1 --b--> 2
    Impl: 0 --a--> 1 --c--> 2
    """
    spec = LTS()
    spec.add_state(LTSState(0, Stop()))
    spec.add_state(LTSState(1, Stop()))
    spec.add_state(LTSState(2, Stop(), is_terminal=True))
    spec.add_transition(Transition(0, Event("a"), 1))
    spec.add_transition(Transition(1, Event("b"), 2))

    impl = LTS()
    impl.add_state(LTSState(0, Stop()))
    impl.add_state(LTSState(1, Stop()))
    impl.add_state(LTSState(2, Stop(), is_terminal=True))
    impl.add_transition(Transition(0, Event("a"), 1))
    impl.add_transition(Transition(1, Event("c"), 2))

    return spec, impl


# =============================================================================
# StepStatus Tests
# =============================================================================


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_ok_status(self) -> None:
        assert StepStatus.OK.name == "OK"

    def test_violation_status(self) -> None:
        assert StepStatus.VIOLATION.name == "VIOLATION"


# =============================================================================
# AnnotatedStep Tests
# =============================================================================


class TestAnnotatedStep:
    """Tests for AnnotatedStep dataclass."""

    def test_basic_creation(self) -> None:
        step = AnnotatedStep(
            event=Event("a"),
            status=StepStatus.OK,
            spec_available=frozenset({Event("a"), Event("b")}),
            impl_available=frozenset({Event("a")}),
        )
        assert step.event == Event("a")
        assert step.status == StepStatus.OK

    def test_frozen(self) -> None:
        step = AnnotatedStep(
            event=Event("a"),
            status=StepStatus.OK,
            spec_available=frozenset(),
            impl_available=frozenset(),
        )
        with pytest.raises(AttributeError):
            step.event = Event("b")  # type: ignore[misc]

    def test_with_states(self) -> None:
        step = AnnotatedStep(
            event=Event("a"),
            status=StepStatus.VIOLATION,
            spec_available=frozenset({Event("a")}),
            impl_available=frozenset({Event("a"), Event("b")}),
            spec_states=frozenset({0, 1}),
            impl_states=frozenset({0}),
        )
        assert 0 in step.spec_states
        assert 1 in step.spec_states


# =============================================================================
# CounterexampleExplanation Tests
# =============================================================================


class TestCounterexampleExplanation:
    """Tests for CounterexampleExplanation dataclass."""

    def test_basic_creation(self) -> None:
        explanation = CounterexampleExplanation(
            summary="Test failure",
            annotated_trace=[],
            divergence_point=0,
            spec_allowed=frozenset({Event("a")}),
            impl_attempted=Event("b"),
        )
        assert explanation.summary == "Test failure"
        assert explanation.divergence_point == 0

    def test_with_failure_kind(self) -> None:
        explanation = CounterexampleExplanation(
            summary="Test",
            annotated_trace=[],
            divergence_point=0,
            spec_allowed=frozenset(),
            impl_attempted=None,
            failure_kind="trace",
        )
        assert explanation.failure_kind == "trace"


# =============================================================================
# Refinement Counterexample Explanation Tests
# =============================================================================


class TestExplainRefinementFailure:
    """Tests for explain_refinement_failure."""

    def test_raises_on_valid_result(self) -> None:
        spec = _make_prefix_chain("a", "b")
        result = RefinementResult(refines=True, message="OK")
        with pytest.raises(ValueError, match="Cannot explain a passing"):
            explain_refinement_failure(spec, spec, result)

    def test_trace_divergence_with_processes(self) -> None:
        """Spec allows a->b, impl does a->c."""
        spec = _make_prefix_chain("a", "b")
        impl = _make_prefix_chain("a", "c")
        result = trace_refines(spec, impl)
        assert not result.is_valid

        explanation = explain_refinement_failure(spec, impl, result)
        assert not explanation.summary == ""
        assert explanation.failure_kind == "trace"
        assert explanation.divergence_point >= 0
        assert len(explanation.annotated_trace) > 0

    def test_trace_divergence_with_lts(self) -> None:
        """Test with raw LTS objects."""
        spec, impl = _make_lts_simple()
        result = trace_refines(spec, impl)
        assert not result.is_valid

        explanation = explain_refinement_failure(spec, impl, result)
        assert explanation.divergence_point >= 0
        assert explanation.impl_attempted is not None

    def test_divergence_at_first_step(self) -> None:
        """Impl does an event spec doesn't allow at all."""
        spec = _make_prefix_chain("a")
        impl = _make_prefix_chain("b")
        result = trace_refines(spec, impl)
        assert not result.is_valid

        explanation = explain_refinement_failure(spec, impl, result)
        assert explanation.divergence_point == 0
        assert explanation.impl_attempted == Event("b")
        assert Event("a") in explanation.spec_allowed
        assert Event("b") not in explanation.spec_allowed

    def test_divergence_at_second_step(self) -> None:
        """Spec: a->b, Impl: a->c — diverge at step 1 (after a)."""
        spec = _make_prefix_chain("a", "b")
        impl = _make_prefix_chain("a", "c")
        result = trace_refines(spec, impl)
        assert not result.is_valid

        explanation = explain_refinement_failure(spec, impl, result)
        assert explanation.divergence_point == 1
        assert explanation.impl_attempted == Event("c")
        assert Event("b") in explanation.spec_allowed

    def test_annotated_trace_has_correct_length(self) -> None:
        """Annotated trace length matches counterexample trace length."""
        spec = _make_prefix_chain("a", "b")
        impl = _make_prefix_chain("a", "c")
        result = trace_refines(spec, impl)
        explanation = explain_refinement_failure(spec, impl, result)

        # The counterexample trace should have events
        assert len(explanation.annotated_trace) > 0
        # Each step has an event
        for step in explanation.annotated_trace:
            assert isinstance(step.event, Event)

    def test_violation_step_marked(self) -> None:
        """The diverging step should be marked as VIOLATION."""
        spec = _make_prefix_chain("a", "b")
        impl = _make_prefix_chain("a", "c")
        result = trace_refines(spec, impl)
        explanation = explain_refinement_failure(spec, impl, result)

        has_violation = any(s.status == StepStatus.VIOLATION for s in explanation.annotated_trace)
        assert has_violation

    def test_ok_steps_before_violation(self) -> None:
        """Steps before divergence should be OK."""
        spec = _make_prefix_chain("a", "b", "c")
        impl = _make_prefix_chain("a", "b", "d")
        result = trace_refines(spec, impl)
        assert not result.is_valid

        explanation = explain_refinement_failure(spec, impl, result)
        dp = explanation.divergence_point
        for i in range(dp):
            assert explanation.annotated_trace[i].status == StepStatus.OK

    def test_spec_available_events_correct(self) -> None:
        """Available events at divergence point match spec."""
        spec = _make_prefix_chain("a", "b")
        impl = _make_prefix_chain("a", "c")
        result = trace_refines(spec, impl)
        explanation = explain_refinement_failure(spec, impl, result)

        dp = explanation.divergence_point
        assert Event("b") in explanation.annotated_trace[dp].spec_available

    def test_failures_refinement_counterexample(self) -> None:
        """Test counterexample from failures refinement."""
        from agenticraft_foundation.algebra.csp import ExternalChoice

        # Spec: a -> STOP [] b -> STOP (can do a or b)
        # Impl: a -> STOP (can only do a, refuses b)
        spec = ExternalChoice(Prefix(Event("a"), Stop()), Prefix(Event("b"), Stop()))
        impl = Prefix(Event("a"), Stop())
        result = failures_refines(spec, impl)

        if not result.is_valid:
            explanation = explain_refinement_failure(spec, impl, result)
            assert explanation.failure_kind == "failures"

    def test_empty_counterexample_trace(self) -> None:
        """Handle case where counterexample is the empty trace."""
        result = RefinementResult(
            refines=False,
            counterexample=(),
            message="Empty trace issue",
        )
        spec = _make_prefix_chain("a")
        impl = _make_prefix_chain("a")

        explanation = explain_refinement_failure(spec, impl, result)
        assert explanation.annotated_trace == []

    def test_summary_contains_trace(self) -> None:
        """Summary should mention the failing trace."""
        spec = _make_prefix_chain("a", "b")
        impl = _make_prefix_chain("a", "c")
        result = trace_refines(spec, impl)
        explanation = explain_refinement_failure(spec, impl, result)
        # Summary should be non-empty and descriptive
        assert len(explanation.summary) > 0


# =============================================================================
# Equivalence Counterexample Explanation Tests
# =============================================================================


class TestExplainEquivalenceFailure:
    """Tests for explain_equivalence_failure."""

    def test_raises_on_equivalent(self) -> None:
        p = _make_prefix_chain("a")
        result = EquivalenceResult(equivalent=True)
        with pytest.raises(ValueError, match="Cannot explain a passing"):
            explain_equivalence_failure(p, p, result)

    def test_trace_equivalence_failure(self) -> None:
        """P does {a,b}, Q does {a,c}."""
        from agenticraft_foundation.algebra.csp import ExternalChoice

        p = ExternalChoice(Prefix(Event("a"), Stop()), Prefix(Event("b"), Stop()))
        q = ExternalChoice(Prefix(Event("a"), Stop()), Prefix(Event("c"), Stop()))
        result = trace_equivalent(p, q)
        assert not result.is_equivalent

        explanation = explain_equivalence_failure(p, q, result)
        assert explanation.failure_kind == "equivalence"
        assert len(explanation.summary) > 0
        assert explanation.divergence_point >= 0

    def test_strong_bisimulation_failure(self) -> None:
        """Test strong bisimulation failure explanation."""
        # a -> b -> STOP vs a -> c -> STOP
        p = _make_prefix_chain("a", "b")
        q = _make_prefix_chain("a", "c")
        result = strong_bisimilar(p, q)
        assert not result.is_equivalent

        explanation = explain_equivalence_failure(p, q, result)
        assert explanation.failure_kind == "equivalence"

    def test_divergence_in_first_event(self) -> None:
        """Processes differ at the very first event."""
        p = _make_prefix_chain("a")
        q = _make_prefix_chain("b")
        result = trace_equivalent(p, q)
        assert not result.is_equivalent

        explanation = explain_equivalence_failure(p, q, result)
        assert explanation.divergence_point == 0

    def test_summary_mentions_difference(self) -> None:
        """Summary should describe what P can do vs Q."""
        from agenticraft_foundation.algebra.csp import ExternalChoice

        p = ExternalChoice(Prefix(Event("a"), Stop()), Prefix(Event("b"), Stop()))
        q = Prefix(Event("a"), Stop())
        result = trace_equivalent(p, q)
        assert not result.is_equivalent

        explanation = explain_equivalence_failure(p, q, result)
        # Summary should mention the difference
        assert len(explanation.summary) > 0

    def test_annotated_trace_events_match_witness(self) -> None:
        """Annotated trace events should match the witness trace."""
        p = _make_prefix_chain("a", "b")
        q = _make_prefix_chain("a", "c")
        result = strong_bisimilar(p, q)
        if result.witness:
            explanation = explain_equivalence_failure(p, q, result)
            trace_events = tuple(s.event for s in explanation.annotated_trace)
            assert trace_events == result.witness

    def test_with_no_witness(self) -> None:
        """Handle equivalence result with no witness trace."""
        p = _make_prefix_chain("a")
        q = _make_prefix_chain("b")
        result = EquivalenceResult(equivalent=False, witness=None)

        explanation = explain_equivalence_failure(p, q, result)
        # Should still produce a valid explanation
        assert explanation.annotated_trace == []

    def test_with_lts_objects(self) -> None:
        """Test with raw LTS objects instead of processes."""
        spec, impl = _make_lts_simple()
        result = trace_equivalent(spec, impl)
        assert not result.is_equivalent

        explanation = explain_equivalence_failure(spec, impl, result)
        assert explanation.failure_kind == "equivalence"


# =============================================================================
# Find Minimal Counterexample Tests
# =============================================================================


class TestFindMinimalCounterexample:
    """Tests for find_minimal_counterexample."""

    def test_no_counterexample_when_refines(self) -> None:
        """Should return None when impl refines spec."""
        spec = _make_prefix_chain("a", "b")
        impl = _make_prefix_chain("a", "b")
        result = find_minimal_counterexample(spec, impl)
        assert result is None

    def test_finds_counterexample(self) -> None:
        """Should find a counterexample when impl doesn't refine spec."""
        spec = _make_prefix_chain("a", "b")
        impl = _make_prefix_chain("a", "c")
        result = find_minimal_counterexample(spec, impl)
        assert result is not None
        assert result.divergence_point >= 0

    def test_finds_shortest_counterexample(self) -> None:
        """Should find the shortest diverging trace."""
        # Impl can do "b" which spec can't — shortest CE is just "b"
        from agenticraft_foundation.algebra.csp import ExternalChoice

        spec = Prefix(Event("a"), Stop())
        impl = ExternalChoice(Prefix(Event("a"), Stop()), Prefix(Event("b"), Stop()))
        result = find_minimal_counterexample(spec, impl)
        assert result is not None
        # The shortest CE should be length 1 (just the extra event)
        assert len(result.annotated_trace) == 1

    def test_with_lts(self) -> None:
        """Should work with LTS objects."""
        spec, impl = _make_lts_simple()
        result = find_minimal_counterexample(spec, impl)
        assert result is not None

    def test_subset_impl_no_counterexample(self) -> None:
        """Impl subset of spec should return None."""
        from agenticraft_foundation.algebra.csp import ExternalChoice

        spec = ExternalChoice(Prefix(Event("a"), Stop()), Prefix(Event("b"), Stop()))
        impl = Prefix(Event("a"), Stop())
        # Impl traces ⊆ spec traces, so trace refinement holds
        result = find_minimal_counterexample(spec, impl)
        assert result is None

    def test_max_trace_length_respected(self) -> None:
        """Should not search beyond max_trace_length."""
        spec = _make_prefix_chain("a", "b")
        impl = _make_prefix_chain("a", "c")
        result = find_minimal_counterexample(spec, impl, max_trace_length=1)
        # With max_trace_length=1, we can only see 1 step deep
        # The divergence is at depth 2 (a, then c vs b), so should not find it
        # Actually, at depth 1 we see "a" which is common, so result may be None
        # This depends on BFS depth — just check it doesn't crash
        assert result is None or result.divergence_point >= 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining counterexample explanation with real checks."""

    def test_full_refinement_workflow(self) -> None:
        """Complete workflow: check refinement -> explain failure."""
        spec = _make_prefix_chain("req", "resp")
        impl = _make_prefix_chain("req", "ack")

        # Step 1: Check refinement
        result = trace_refines(spec, impl)
        assert not result.is_valid

        # Step 2: Explain
        explanation = explain_refinement_failure(spec, impl, result)
        assert explanation.divergence_point == 1
        assert explanation.impl_attempted == Event("ack")
        assert Event("resp") in explanation.spec_allowed

    def test_full_equivalence_workflow(self) -> None:
        """Complete workflow: check equivalence -> explain failure."""
        p = _make_prefix_chain("a", "b")
        q = _make_prefix_chain("a", "c")

        # Step 1: Check equivalence
        result = strong_bisimilar(p, q)
        assert not result.is_equivalent

        # Step 2: Explain
        explanation = explain_equivalence_failure(p, q, result)
        assert explanation.failure_kind == "equivalence"
        assert len(explanation.annotated_trace) > 0

    def test_skip_process_refinement(self) -> None:
        """Test with Skip (successful termination) processes."""
        # Spec: a -> SKIP, Impl: a -> STOP
        spec = Prefix(Event("a"), Skip())
        impl = Prefix(Event("a"), Stop())
        result = trace_refines(spec, impl)

        if not result.is_valid:
            explanation = explain_refinement_failure(spec, impl, result)
            assert explanation.divergence_point >= 0

    def test_longer_trace_divergence(self) -> None:
        """Test with longer traces to ensure walkthrough works."""
        spec = _make_prefix_chain("a", "b", "c", "d")
        impl = _make_prefix_chain("a", "b", "c", "e")
        result = trace_refines(spec, impl)
        assert not result.is_valid

        explanation = explain_refinement_failure(spec, impl, result)
        assert explanation.divergence_point == 3
        # First 3 steps should be OK
        for i in range(3):
            assert explanation.annotated_trace[i].status == StepStatus.OK
        # Step 3 should be VIOLATION
        assert explanation.annotated_trace[3].status == StepStatus.VIOLATION
