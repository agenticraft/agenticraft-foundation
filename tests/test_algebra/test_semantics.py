"""Tests for LTS operational semantics."""

from __future__ import annotations

from agenticraft_foundation.algebra import (
    TICK,
    Event,
    accepts,
    analyze_liveness,
    build_lts,
    choice,
    detect_deadlock,
    is_deadlock_free,
    maximal_traces,
    parallel,
    prefix,
    sequential,
    skip,
    stop,
    traces,
)


class TestLTSBuilder:
    """Tests for LTS construction."""

    def test_build_skip(self):
        """Test building LTS for SKIP."""
        lts = build_lts(skip())
        assert lts.num_states >= 1
        assert TICK in lts.alphabet

    def test_build_stop(self):
        """Test building LTS for STOP."""
        lts = build_lts(stop())
        assert lts.num_states == 1
        assert len(lts.transitions) == 0

    def test_build_prefix(self):
        """Test building LTS for prefix."""
        lts = build_lts(prefix("a", skip()))
        assert Event("a") in lts.alphabet
        assert lts.num_transitions >= 1

    def test_build_choice(self):
        """Test building LTS for choice."""
        p = choice(prefix("a", skip()), prefix("b", skip()))
        lts = build_lts(p)
        assert Event("a") in lts.alphabet
        assert Event("b") in lts.alphabet

    def test_build_parallel(self):
        """Test building LTS for parallel."""
        p = parallel(prefix("a", skip()), prefix("b", skip()))
        lts = build_lts(p)
        assert Event("a") in lts.alphabet
        assert Event("b") in lts.alphabet


class TestTraces:
    """Tests for trace generation."""

    def test_traces_skip(self):
        """Test traces of SKIP."""
        lts = build_lts(skip())
        trace_set = set(traces(lts))
        assert () in trace_set  # Empty trace
        assert (TICK,) in trace_set

    def test_traces_prefix(self):
        """Test traces of prefix."""
        lts = build_lts(prefix("a", skip()))
        trace_set = set(traces(lts))
        assert () in trace_set
        assert (Event("a"),) in trace_set
        assert (Event("a"), TICK) in trace_set

    def test_traces_choice(self):
        """Test traces of choice."""
        p = choice(prefix("a", skip()), prefix("b", skip()))
        lts = build_lts(p)
        trace_set = set(traces(lts))
        assert (Event("a"),) in trace_set
        assert (Event("b"),) in trace_set

    def test_traces_sequential(self):
        """Test traces of sequential."""
        p = sequential(prefix("a", skip()), prefix("b", skip()))
        lts = build_lts(p)
        trace_set = set(traces(lts))
        assert (Event("a"),) in trace_set
        # After a, TICK from first enables b
        assert (Event("a"), Event("b")) in trace_set


class TestMaximalTraces:
    """Tests for maximal trace generation."""

    def test_maximal_traces_skip(self):
        """Test maximal traces of SKIP."""
        lts = build_lts(skip())
        max_traces = list(maximal_traces(lts))
        # maximal_traces filters out TICK by default
        # SKIP terminates, producing an empty maximal trace
        assert () in max_traces

    def test_maximal_traces_stop(self):
        """Test maximal traces of STOP."""
        lts = build_lts(stop())
        max_traces = list(maximal_traces(lts))
        # STOP deadlocks immediately with empty trace
        assert () in max_traces


class TestAccepts:
    """Tests for trace acceptance."""

    def test_accepts_empty(self):
        """Test accepting empty trace."""
        lts = build_lts(prefix("a", skip()))
        assert accepts(lts, ())

    def test_accepts_valid_trace(self):
        """Test accepting valid trace."""
        lts = build_lts(prefix("a", skip()))
        assert accepts(lts, (Event("a"),))

    def test_rejects_invalid_trace(self):
        """Test rejecting invalid trace."""
        lts = build_lts(prefix("a", skip()))
        assert not accepts(lts, (Event("b"),))


class TestDeadlockDetection:
    """Tests for deadlock detection."""

    def test_stop_has_deadlock(self):
        """Test STOP is detected as deadlock."""
        lts = build_lts(stop())
        result = detect_deadlock(lts)
        assert result.has_deadlock

    def test_skip_no_deadlock(self):
        """Test SKIP has no deadlock."""
        lts = build_lts(skip())
        result = detect_deadlock(lts)
        assert not result.has_deadlock

    def test_prefix_skip_no_deadlock(self):
        """Test a → SKIP has no deadlock."""
        lts = build_lts(prefix("a", skip()))
        result = detect_deadlock(lts)
        assert not result.has_deadlock

    def test_deadlock_trace(self):
        """Test deadlock trace is returned."""
        # choice where one branch deadlocks
        p = choice(prefix("a", stop()), prefix("b", skip()))
        lts = build_lts(p)
        result = detect_deadlock(lts)
        assert result.has_deadlock
        assert len(result.deadlock_traces) > 0


class TestLivenessAnalysis:
    """Tests for liveness analysis."""

    def test_live_process(self):
        """Test a live process."""
        p = prefix("a", skip())
        lts = build_lts(p)
        result = analyze_liveness(lts, {Event("a")})
        assert result.live_events[Event("a")]

    def test_stop_is_live_with_no_events(self):
        """Test STOP with no events of interest is considered live."""
        lts = build_lts(stop())
        # With no events of interest, is_live checks for stuck states
        # STOP has no transitions, but it's the only state - no stuck detection
        result = analyze_liveness(lts)
        # STOP is technically "live" when there are no events to be live for
        # It just deadlocks immediately
        assert len(result.live_events) == 0


class TestIsDeadlockFree:
    """Tests for deadlock freedom check."""

    def test_skip_deadlock_free(self):
        """Test SKIP is deadlock free."""
        assert is_deadlock_free(skip())

    def test_stop_not_deadlock_free(self):
        """Test STOP is not deadlock free."""
        assert not is_deadlock_free(stop())

    def test_prefix_skip_deadlock_free(self):
        """Test a → SKIP is deadlock free."""
        assert is_deadlock_free(prefix("a", skip()))

    def test_choice_with_deadlock_not_free(self):
        """Test choice with deadlock branch is not deadlock free."""
        p = choice(prefix("a", stop()), prefix("b", skip()))
        assert not is_deadlock_free(p)
