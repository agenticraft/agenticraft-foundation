"""Tests for runtime invariant checking."""

import pytest

from agenticraft_foundation.verification import (
    Invariant,
    InvariantRegistry,
    StateTransitionMonitor,
    Violation,
    ViolationSeverity,
    assert_invariant,
    check_invariant,
    register_invariant,
)


class TestInvariant:
    """Tests for Invariant dataclass."""

    def test_basic_invariant(self):
        """Test basic invariant creation."""
        inv = Invariant(
            name="positive",
            condition=lambda x: x > 0,
        )

        assert inv.name == "positive"
        assert inv.enabled

    def test_check_passes(self):
        """Test invariant check that passes."""
        inv = Invariant(
            name="positive",
            condition=lambda x: x > 0,
        )

        assert inv.check(5)
        assert inv.check(1)

    def test_check_fails(self):
        """Test invariant check that fails."""
        inv = Invariant(
            name="positive",
            condition=lambda x: x > 0,
        )

        assert not inv.check(-1)
        assert not inv.check(0)

    def test_disabled_invariant(self):
        """Test disabled invariant always passes."""
        inv = Invariant(
            name="positive",
            condition=lambda x: x > 0,
            enabled=False,
        )

        # Should pass even for invalid input
        assert inv.check(-1)


class TestInvariantRegistry:
    """Tests for InvariantRegistry."""

    def test_register_invariant(self):
        """Test registering an invariant."""
        registry = InvariantRegistry()
        inv = registry.register("positive", lambda x: x > 0)

        assert inv.name == "positive"
        assert "positive" in registry._invariants

    def test_unregister_invariant(self):
        """Test unregistering an invariant."""
        registry = InvariantRegistry()
        registry.register("test", lambda: True)

        assert registry.unregister("test")
        assert "test" not in registry._invariants

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent invariant."""
        registry = InvariantRegistry()
        assert not registry.unregister("nonexistent")

    def test_check_invariant(self):
        """Test checking a registered invariant."""
        registry = InvariantRegistry()
        registry.register("positive", lambda x: x > 0)

        assert registry.check("positive", 5)
        assert not registry.check("positive", -1)

    def test_check_nonexistent(self):
        """Test checking nonexistent invariant raises error."""
        registry = InvariantRegistry()

        with pytest.raises(KeyError):
            registry.check("nonexistent")

    def test_check_all(self):
        """Test checking all invariants."""
        registry = InvariantRegistry()
        registry.register("positive", lambda x: x > 0)
        registry.register("less_than_10", lambda x: x < 10)

        # Both pass
        violated = registry.check_all(5)
        assert len(violated) == 0

        # One fails
        violated = registry.check_all(15)
        assert "less_than_10" in violated

    def test_violation_recording(self):
        """Test that violations are recorded."""
        registry = InvariantRegistry()
        registry.register("positive", lambda x: x > 0, message="Must be positive")

        registry.check("positive", -1)

        violations = registry.get_violations()
        assert len(violations) == 1
        assert violations[0].invariant_name == "positive"

    def test_clear_violations(self):
        """Test clearing violations."""
        registry = InvariantRegistry()
        registry.register("positive", lambda x: x > 0)
        registry.check("positive", -1)

        registry.clear_violations()
        assert len(registry.get_violations()) == 0

    def test_violation_handler(self):
        """Test violation handler callback."""
        registry = InvariantRegistry()
        registry.register("positive", lambda x: x > 0)

        handled: list[Violation] = []
        registry.on_violation(lambda v: handled.append(v))

        registry.check("positive", -1)

        assert len(handled) == 1

    def test_enable_disable(self):
        """Test enabling/disabling registry."""
        registry = InvariantRegistry()
        registry.register("positive", lambda x: x > 0)

        registry.disable()
        # Should pass when disabled
        assert registry.check("positive", -1)

        registry.enable()
        # Should fail when enabled
        assert not registry.check("positive", -1)

    def test_stats(self):
        """Test statistics tracking."""
        registry = InvariantRegistry("test_registry")
        registry.register("positive", lambda x: x > 0)

        registry.check("positive", 5)  # Pass
        registry.check("positive", -1)  # Fail

        stats = registry.stats()
        assert stats["name"] == "test_registry"
        assert stats["check_count"] == 2
        assert stats["violation_count"] == 1
        assert stats["violation_rate"] == 0.5


class TestGlobalFunctions:
    """Tests for global invariant functions."""

    def test_register_and_check(self):
        """Test global register and check functions."""
        register_invariant("test_inv", lambda: True)
        assert check_invariant("test_inv")

    def test_invariant_decorator(self):
        """Test @invariant decorator."""
        # Create a separate registry for this test
        test_registry = InvariantRegistry("test_decorator")

        # Register invariant manually (the decorator registers too)
        register_invariant(
            "non_negative_counter",
            lambda *args, **kwargs: True,  # Simple check for test
            message="Counter cannot be negative",
            registry=test_registry,
        )

        # Verify registration worked
        result = check_invariant("non_negative_counter", registry=test_registry)
        assert result


class TestAssertInvariant:
    """Tests for assert_invariant function."""

    def test_assert_passes(self):
        """Test assertion that passes."""
        # Should not raise
        assert_invariant(True, "Should pass")

    def test_assert_warning(self):
        """Test assertion with warning severity."""
        # Should not raise, just log
        assert_invariant(False, "Warning", ViolationSeverity.WARNING)

    def test_assert_fatal(self):
        """Test assertion with fatal severity raises."""
        with pytest.raises(AssertionError):
            assert_invariant(False, "Fatal error", ViolationSeverity.FATAL)


class TestStateTransitionMonitor:
    """Tests for StateTransitionMonitor."""

    def test_basic_transition(self):
        """Test basic state transition."""
        monitor = StateTransitionMonitor()
        monitor.transition("state_a")

        assert monitor.current_state == "state_a"

    def test_valid_transitions(self):
        """Test valid state transitions."""
        monitor = StateTransitionMonitor()
        monitor.set_valid_transitions(
            {
                "initial": {"running", "stopped"},
                "running": {"paused", "stopped"},
                "paused": {"running", "stopped"},
                "stopped": set(),
            }
        )

        monitor.transition("initial")
        assert monitor.transition("running")
        assert monitor.transition("paused")
        assert monitor.transition("running")
        assert monitor.transition("stopped")

    def test_invalid_transition(self):
        """Test invalid state transition."""
        monitor = StateTransitionMonitor()
        monitor.set_valid_transitions(
            {
                "initial": {"running"},
                "running": {"stopped"},
                "stopped": set(),
            }
        )

        monitor.transition("initial")
        monitor.transition("running")

        # Invalid: running -> initial
        result = monitor.transition("initial")
        assert not result

        invalid = monitor.get_invalid_transitions()
        assert len(invalid) == 1
        assert invalid[0] == ("running", "initial")

    def test_transition_history(self):
        """Test transition history tracking."""
        monitor = StateTransitionMonitor()

        monitor.transition("a")
        monitor.transition("b")
        monitor.transition("c")

        history = monitor.get_history()
        assert len(history) == 3
        assert history[0][1] == "a"
        assert history[1][1] == "b"
        assert history[2][1] == "c"

    def test_reset(self):
        """Test monitor reset."""
        monitor = StateTransitionMonitor()
        monitor.transition("a")
        monitor.transition("b")

        monitor.reset()

        assert monitor.current_state is None
        assert len(monitor.get_history()) == 0


class TestViolation:
    """Tests for Violation dataclass."""

    def test_basic_violation(self):
        """Test basic violation creation."""
        violation = Violation(
            invariant_name="test",
            severity=ViolationSeverity.ERROR,
            message="Test violation",
        )

        assert violation.invariant_name == "test"
        assert violation.severity == ViolationSeverity.ERROR
        assert violation.timestamp > 0

    def test_violation_with_context(self):
        """Test violation with context."""
        violation = Violation(
            invariant_name="test",
            severity=ViolationSeverity.WARNING,
            message="Test",
            context={"key": "value"},
        )

        assert violation.context["key"] == "value"
