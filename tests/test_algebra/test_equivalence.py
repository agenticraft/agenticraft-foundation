"""Tests for process equivalence checking."""

from __future__ import annotations

import pytest

from agenticraft_foundation.algebra import (
    are_equivalent,
    choice,
    failures_equivalent,
    prefix,
    skip,
    stop,
    strong_bisimilar,
    trace_equivalent,
    weak_bisimilar,
)


class TestTraceEquivalence:
    """Tests for trace equivalence."""

    def test_same_process_equivalent(self):
        """Test that a process is trace equivalent to itself."""
        p = prefix("a", skip())
        result = trace_equivalent(p, p)
        assert result.is_equivalent

    def test_structurally_equal_equivalent(self):
        """Test structurally equal processes are equivalent."""
        p1 = prefix("a", skip())
        p2 = prefix("a", skip())
        result = trace_equivalent(p1, p2)
        assert result.is_equivalent

    def test_different_events_not_equivalent(self):
        """Test processes with different events are not equivalent."""
        p1 = prefix("a", skip())
        p2 = prefix("b", skip())
        result = trace_equivalent(p1, p2)
        assert not result.is_equivalent
        assert result.witness is not None

    def test_choice_order_equivalent(self):
        """Test choice is trace equivalent regardless of order."""
        p1 = choice(prefix("a", skip()), prefix("b", skip()))
        p2 = choice(prefix("b", skip()), prefix("a", skip()))
        result = trace_equivalent(p1, p2)
        assert result.is_equivalent

    def test_skip_stop_not_equivalent(self):
        """Test SKIP and STOP are not trace equivalent."""
        result = trace_equivalent(skip(), stop())
        assert not result.is_equivalent


class TestStrongBisimulation:
    """Tests for strong bisimulation."""

    def test_same_process_bisimilar(self):
        """Test a process is strongly bisimilar to itself."""
        p = prefix("a", skip())
        result = strong_bisimilar(p, p)
        assert result.is_equivalent

    def test_structurally_equal_bisimilar(self):
        """Test structurally equal processes are bisimilar."""
        p1 = prefix("a", prefix("b", skip()))
        p2 = prefix("a", prefix("b", skip()))
        result = strong_bisimilar(p1, p2)
        assert result.is_equivalent

    def test_different_structure_not_bisimilar(self):
        """Test differently structured processes are not bisimilar."""
        p1 = prefix("a", skip())
        p2 = prefix("b", skip())
        result = strong_bisimilar(p1, p2)
        assert not result.is_equivalent

    def test_choice_order_bisimilar(self):
        """Test choice is bisimilar regardless of order."""
        p1 = choice(prefix("a", skip()), prefix("b", skip()))
        p2 = choice(prefix("b", skip()), prefix("a", skip()))
        result = strong_bisimilar(p1, p2)
        assert result.is_equivalent

    def test_bisimulation_relation_returned(self):
        """Test bisimulation relation is returned when equivalent."""
        p = prefix("a", skip())
        result = strong_bisimilar(p, p)
        assert result.is_equivalent
        assert result.relation is not None


class TestWeakBisimulation:
    """Tests for weak bisimulation."""

    def test_same_process_weak_bisimilar(self):
        """Test a process is weakly bisimilar to itself."""
        p = prefix("a", skip())
        result = weak_bisimilar(p, p)
        assert result.is_equivalent

    def test_tau_hidden_equivalent(self):
        """Test processes equal modulo τ are weakly bisimilar."""
        # P = a → SKIP
        p1 = prefix("a", skip())
        # Q = (a → SKIP) with nothing hidden
        p2 = prefix("a", skip())
        result = weak_bisimilar(p1, p2)
        assert result.is_equivalent


class TestFailuresEquivalence:
    """Tests for failures equivalence."""

    def test_same_process_failures_equivalent(self):
        """Test a process is failures equivalent to itself."""
        p = prefix("a", skip())
        result = failures_equivalent(p, p)
        assert result.is_equivalent

    def test_different_refusals_not_equivalent(self):
        """Test processes with different refusals are not equivalent."""
        # STOP refuses everything
        # a → SKIP initially only refuses b, c, etc.
        result = failures_equivalent(stop(), prefix("a", skip()))
        assert not result.is_equivalent


class TestAreEquivalent:
    """Tests for convenience equivalence function."""

    def test_are_equivalent_trace(self):
        """Test are_equivalent with trace mode."""
        p1 = prefix("a", skip())
        p2 = prefix("a", skip())
        assert are_equivalent(p1, p2, "trace")

    def test_are_equivalent_strong(self):
        """Test are_equivalent with strong bisimulation."""
        p1 = prefix("a", skip())
        p2 = prefix("a", skip())
        assert are_equivalent(p1, p2, "strong")

    def test_are_equivalent_weak(self):
        """Test are_equivalent with weak bisimulation."""
        p1 = prefix("a", skip())
        p2 = prefix("a", skip())
        assert are_equivalent(p1, p2, "weak")

    def test_are_equivalent_failures(self):
        """Test are_equivalent with failures mode."""
        p1 = prefix("a", skip())
        p2 = prefix("a", skip())
        assert are_equivalent(p1, p2, "failures")

    def test_invalid_mode_raises(self):
        """Test invalid mode raises error."""
        p = prefix("a", skip())
        with pytest.raises(ValueError):
            are_equivalent(p, p, "invalid")
