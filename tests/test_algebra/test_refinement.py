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
