"""Tests for refinement checking."""

from __future__ import annotations

import pytest

from agenticraft_foundation.algebra import (
    analyze_refinement,
    check_divergence_free,
    choice,
    failures_refines,
    fd_refines,
    prefix,
    refines,
    skip,
    stop,
    trace_refines,
)
from agenticraft_foundation.algebra.csp import TAU, Event
from agenticraft_foundation.algebra.refinement import (
    _find_trace_to_state,
    _has_tau_cycle,
    _trace_str,
)
from agenticraft_foundation.algebra.semantics import LTS, LTSState, Transition


class TestTraceRefinement:
    """Tests for trace refinement."""

    def test_same_process_refines(self):
        """Test a process trace-refines itself."""
        p = prefix("a", skip())
        result = trace_refines(p, p)
        assert result.is_valid

    def test_subset_traces_refines(self):
        """Test process with subset of traces refines."""
        # SPEC: a → SKIP or b → SKIP
        spec = choice(prefix("a", skip()), prefix("b", skip()))
        # IMPL: just a → SKIP (subset of traces)
        impl = prefix("a", skip())
        result = trace_refines(spec, impl)
        assert result.is_valid

    def test_extra_traces_not_refines(self):
        """Test process with extra traces does not refine."""
        # SPEC: just a → SKIP
        spec = prefix("a", skip())
        # IMPL: a → SKIP or b → SKIP (extra trace)
        impl = choice(prefix("a", skip()), prefix("b", skip()))
        result = trace_refines(spec, impl)
        assert not result.is_valid
        assert result.counterexample is not None

    def test_stop_refines_anything(self):
        """Test STOP refines any process (empty traces)."""
        spec = choice(prefix("a", skip()), prefix("b", skip()))
        result = trace_refines(spec, stop())
        assert result.is_valid


class TestFailuresRefinement:
    """Tests for failures refinement."""

    def test_same_process_refines(self):
        """Test a process failures-refines itself."""
        p = prefix("a", skip())
        result = failures_refines(p, p)
        assert result.is_valid

    def test_extra_trace_not_refines(self):
        """Test process with extra traces does not refine."""
        spec = prefix("a", skip())
        impl = choice(prefix("a", skip()), prefix("b", skip()))
        result = failures_refines(spec, impl)
        assert not result.is_valid


class TestDivergenceFreedom:
    """Tests for divergence freedom."""

    def test_skip_divergence_free(self):
        """Test SKIP is divergence-free."""
        result = check_divergence_free(skip())
        assert result.is_divergence_free

    def test_stop_divergence_free(self):
        """Test STOP is divergence-free (no progress, not divergence)."""
        result = check_divergence_free(stop())
        assert result.is_divergence_free

    def test_prefix_divergence_free(self):
        """Test a → SKIP is divergence-free."""
        result = check_divergence_free(prefix("a", skip()))
        assert result.is_divergence_free


class TestFDRefinement:
    """Tests for failures-divergences refinement."""

    def test_same_process_fd_refines(self):
        """Test a process FD-refines itself."""
        p = prefix("a", skip())
        result = fd_refines(p, p)
        assert result.is_valid

    def test_extra_traces_not_fd_refines(self):
        """Test process with extra traces does not FD-refine."""
        spec = prefix("a", skip())
        impl = choice(prefix("a", skip()), prefix("b", skip()))
        result = fd_refines(spec, impl)
        assert not result.is_valid


class TestAnalyzeRefinement:
    """Tests for comprehensive refinement analysis."""

    def test_analyze_valid_refinement(self):
        """Test analyzing a valid refinement."""
        spec = choice(prefix("a", skip()), prefix("b", skip()))
        impl = prefix("a", skip())
        report = analyze_refinement(spec, impl)
        assert report.overall_result

    def test_analyze_invalid_refinement(self):
        """Test analyzing an invalid refinement."""
        spec = prefix("a", skip())
        impl = choice(prefix("a", skip()), prefix("b", skip()))
        report = analyze_refinement(spec, impl)
        assert not report.overall_result

    def test_analyze_includes_samples(self):
        """Test analysis includes trace samples."""
        spec = prefix("a", skip())
        impl = prefix("a", skip())
        report = analyze_refinement(spec, impl)
        assert len(report.spec_traces_sample) > 0
        assert len(report.impl_traces_sample) > 0


class TestRefinesConvenience:
    """Tests for convenience refines function."""

    def test_refines_trace(self):
        """Test refines with trace mode."""
        spec = choice(prefix("a", skip()), prefix("b", skip()))
        impl = prefix("a", skip())
        assert refines(spec, impl, "trace")

    def test_refines_failures(self):
        """Test refines with failures mode."""
        p = prefix("a", skip())
        assert refines(p, p, "failures")

    def test_refines_fd(self):
        """Test refines with fd mode."""
        p = prefix("a", skip())
        assert refines(p, p, "fd")

    def test_invalid_mode_raises(self):
        """Test invalid mode raises error."""
        p = prefix("a", skip())
        with pytest.raises(ValueError):
            refines(p, p, "invalid")


class TestTraceStr:
    """Tests for _trace_str helper."""

    def test_empty_trace(self):
        """Test empty trace formatting."""
        assert _trace_str(()) == "⟨⟩"

    def test_nonempty_trace(self):
        """Test non-empty trace formatting."""
        trace = (Event("a"), Event("b"))
        result = _trace_str(trace)
        assert "a" in result
        assert "b" in result


class TestFailuresRefinesEdgeCases:
    """Tests for failures_refines edge paths."""

    def test_failures_refines_trace_not_in_spec(self):
        """Test failure when impl trace is not in spec at all."""
        spec = prefix("a", skip())
        impl = prefix("b", skip())
        result = failures_refines(spec, impl)
        assert not result.is_valid
        assert "not in spec" in result.message

    def test_failures_refines_impl_subset_of_spec(self):
        """Test failures refinement passes when impl refusals are subset of spec."""
        # SPEC: a → (b → SKIP [] c → SKIP) — after <a>, offers b and c
        spec = prefix("a", choice(prefix("b", skip()), prefix("c", skip())))
        # IMPL: same process
        impl = prefix("a", choice(prefix("b", skip()), prefix("c", skip())))
        result = failures_refines(spec, impl)
        assert result.is_valid


def _make_lts_state(state_id: int) -> LTSState:
    """Create a dummy LTSState for testing."""
    return LTSState(id=state_id, process=stop())


class TestDivergenceDetection:
    """Tests for divergence (tau cycles) detection."""

    def test_tau_cycle_detected_via_lts(self):
        """Test _has_tau_cycle finds a tau-cycle in a manually-built LTS."""
        lts = LTS(initial_state=0)
        lts.add_state(_make_lts_state(0))
        lts.add_state(_make_lts_state(1))
        lts.add_transition(Transition(source=0, event=TAU, target=1))
        lts.add_transition(Transition(source=1, event=TAU, target=0))
        assert _has_tau_cycle(lts, 0)
        assert _has_tau_cycle(lts, 1)

    def test_no_tau_cycle(self):
        """Test _has_tau_cycle returns False when no cycle exists."""
        lts = LTS(initial_state=0)
        lts.add_state(_make_lts_state(0))
        lts.add_state(_make_lts_state(1))
        lts.add_transition(Transition(source=0, event=TAU, target=1))
        assert not _has_tau_cycle(lts, 0)

    def test_find_trace_to_state(self):
        """Test _find_trace_to_state finds trace via BFS."""
        a = Event("a")
        lts = LTS(initial_state=0)
        lts.add_state(_make_lts_state(0))
        lts.add_state(_make_lts_state(1))
        lts.add_state(_make_lts_state(2))
        lts.add_transition(Transition(source=0, event=a, target=1))
        lts.add_transition(Transition(source=1, event=TAU, target=2))
        trace = _find_trace_to_state(lts, 2)
        assert trace is not None
        assert trace == (a,)

    def test_find_trace_to_state_initial(self):
        """Test _find_trace_to_state returns empty trace for initial state."""
        lts = LTS(initial_state=0)
        lts.add_state(_make_lts_state(0))
        trace = _find_trace_to_state(lts, 0)
        assert trace == ()

    def test_find_trace_to_state_unreachable(self):
        """Test _find_trace_to_state returns None for unreachable state."""
        lts = LTS(initial_state=0)
        lts.add_state(_make_lts_state(0))
        lts.add_state(_make_lts_state(1))
        trace = _find_trace_to_state(lts, 1)
        assert trace is None

    def test_check_divergence_free_with_tau_cycle_lts(self):
        """Test check_divergence_free on LTS with tau cycle."""
        lts = LTS(initial_state=0)
        lts.add_state(_make_lts_state(0))
        lts.add_state(_make_lts_state(1))
        lts.add_transition(Transition(source=0, event=TAU, target=1))
        lts.add_transition(Transition(source=1, event=TAU, target=0))
        result = check_divergence_free(lts)
        assert not result.is_divergence_free
        assert len(result.divergent_states) > 0


class TestFDRefinesDivergence:
    """Tests for fd_refines with divergence."""

    def test_fd_refines_impl_diverges_spec_doesnt(self):
        """Test FD refinement fails when impl diverges but spec doesn't."""
        spec_lts = LTS(initial_state=0)
        spec_lts.add_state(_make_lts_state(0))
        spec_lts.add_state(_make_lts_state(1))
        spec_lts.add_transition(Transition(source=0, event=Event("a"), target=1))

        impl_lts = LTS(initial_state=0)
        impl_lts.add_state(_make_lts_state(0))
        impl_lts.add_state(_make_lts_state(1))
        impl_lts.add_transition(Transition(source=0, event=TAU, target=1))
        impl_lts.add_transition(Transition(source=1, event=TAU, target=0))

        result = fd_refines(spec_lts, impl_lts)
        assert not result.is_valid
        assert "diverges" in result.message


class TestRefinementReportEdgeCases:
    """Tests for RefinementReport edge cases."""

    def test_overall_result_with_failures_refinement_failing(self):
        """Test overall_result when failures refinement fails."""
        spec = prefix("a", prefix("b", skip()))
        impl = prefix("a", stop())
        report = analyze_refinement(spec, impl, check_failures=True)
        # Trace refinement passes (impl traces subset of spec traces),
        # but failures refinement should fail
        if report.failures_refinement and not report.failures_refinement.is_valid:
            assert not report.overall_result

    def test_overall_result_without_failures(self):
        """Test overall_result when failures check is disabled."""
        spec = choice(prefix("a", skip()), prefix("b", skip()))
        impl = prefix("a", skip())
        report = analyze_refinement(spec, impl, check_failures=False)
        assert report.overall_result
        assert report.failures_refinement is None

    def test_analyze_extra_traces_populated(self):
        """Test that impl_extra_traces is populated for invalid refinement."""
        spec = prefix("a", skip())
        impl = choice(prefix("a", skip()), prefix("b", skip()))
        report = analyze_refinement(spec, impl)
        assert len(report.impl_extra_traces) > 0
