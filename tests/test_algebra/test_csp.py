"""Tests for CSP process primitives."""

from __future__ import annotations

import pytest

from agenticraft_foundation.algebra import (
    TAU,
    TICK,
    Event,
    ExternalChoice,
    Hiding,
    InternalChoice,
    Parallel,
    Prefix,
    ProcessKind,
    Recursion,
    Sequential,
    Skip,
    Stop,
    choice,
    hide,
    interleave,
    internal_choice,
    parallel,
    prefix,
    rec,
    sequential,
    skip,
    stop,
    substitute,
    var,
)


class TestEvent:
    """Tests for Event type."""

    def test_event_creation(self):
        """Test creating events."""
        e = Event("hello")
        assert e == "hello"
        assert str(e) == "hello"

    def test_tick_event(self):
        """Test the TICK special event."""
        assert TICK == "✓"

    def test_tau_event(self):
        """Test the TAU special event."""
        assert TAU == "τ"


class TestStop:
    """Tests for STOP process."""

    def test_stop_kind(self):
        """Test STOP has correct kind."""
        s = Stop()
        assert s.kind == ProcessKind.STOP

    def test_stop_alphabet(self):
        """Test STOP has empty alphabet."""
        s = Stop()
        assert s.alphabet() == frozenset()

    def test_stop_initials(self):
        """Test STOP has no initial events."""
        s = Stop()
        assert s.initials() == frozenset()

    def test_stop_is_deadlocked(self):
        """Test STOP is deadlocked."""
        s = Stop()
        assert s.is_deadlocked()

    def test_stop_cannot_perform(self):
        """Test STOP cannot perform any event."""
        s = Stop()
        with pytest.raises(ValueError):
            s.after(Event("a"))


class TestSkip:
    """Tests for SKIP process."""

    def test_skip_kind(self):
        """Test SKIP has correct kind."""
        s = Skip()
        assert s.kind == ProcessKind.SKIP

    def test_skip_alphabet(self):
        """Test SKIP alphabet is {✓}."""
        s = Skip()
        assert s.alphabet() == frozenset({TICK})

    def test_skip_initials(self):
        """Test SKIP initials is {✓}."""
        s = Skip()
        assert s.initials() == frozenset({TICK})

    def test_skip_is_terminated(self):
        """Test SKIP is terminated."""
        s = Skip()
        assert s.is_terminated()

    def test_skip_after_tick(self):
        """Test SKIP after ✓ becomes STOP."""
        s = Skip()
        result = s.after(TICK)
        assert isinstance(result, Stop)


class TestPrefix:
    """Tests for Prefix process."""

    def test_prefix_creation(self):
        """Test creating prefix process."""
        p = prefix("a", skip())
        assert isinstance(p, Prefix)
        assert p.event == Event("a")

    def test_prefix_kind(self):
        """Test prefix has correct kind."""
        p = prefix("a", skip())
        assert p.kind == ProcessKind.PREFIX

    def test_prefix_alphabet(self):
        """Test prefix alphabet includes event and continuation."""
        p = prefix("a", prefix("b", skip()))
        assert Event("a") in p.alphabet()
        assert Event("b") in p.alphabet()
        assert TICK in p.alphabet()

    def test_prefix_initials(self):
        """Test prefix initials is just the event."""
        p = prefix("a", skip())
        assert p.initials() == frozenset({Event("a")})

    def test_prefix_after(self):
        """Test prefix after event returns continuation."""
        cont = skip()
        p = prefix("a", cont)
        result = p.after(Event("a"))
        assert result is cont

    def test_prefix_wrong_event_raises(self):
        """Test performing wrong event raises error."""
        p = prefix("a", skip())
        with pytest.raises(ValueError):
            p.after(Event("b"))


class TestExternalChoice:
    """Tests for External Choice."""

    def test_choice_creation(self):
        """Test creating external choice."""
        c = choice(prefix("a", skip()), prefix("b", skip()))
        assert isinstance(c, ExternalChoice)

    def test_choice_kind(self):
        """Test choice has correct kind."""
        c = choice(prefix("a", skip()), prefix("b", skip()))
        assert c.kind == ProcessKind.EXTERNAL_CHOICE

    def test_choice_initials(self):
        """Test choice initials is union of both sides."""
        c = choice(prefix("a", skip()), prefix("b", skip()))
        assert c.initials() == frozenset({Event("a"), Event("b")})

    def test_choice_after_left(self):
        """Test choosing left option."""
        left = prefix("a", skip())
        right = prefix("b", skip())
        c = choice(left, right)
        result = c.after(Event("a"))
        assert isinstance(result, Skip)

    def test_choice_after_right(self):
        """Test choosing right option."""
        left = prefix("a", skip())
        right = prefix("b", skip())
        c = choice(left, right)
        result = c.after(Event("b"))
        assert isinstance(result, Skip)


class TestInternalChoice:
    """Tests for Internal Choice."""

    def test_internal_choice_creation(self):
        """Test creating internal choice."""
        c = internal_choice(prefix("a", skip()), prefix("b", skip()))
        assert isinstance(c, InternalChoice)

    def test_internal_choice_includes_tau(self):
        """Test internal choice initials include τ."""
        c = internal_choice(prefix("a", skip()), prefix("b", skip()))
        assert TAU in c.initials()

    def test_internal_choice_after_tau(self):
        """Test τ resolves internal choice."""
        left = prefix("a", skip())
        c = internal_choice(left, prefix("b", skip()))
        result = c.after(TAU)
        assert result is left


class TestParallel:
    """Tests for Parallel composition."""

    def test_interleave_creation(self):
        """Test creating interleaved parallel."""
        p = interleave(prefix("a", skip()), prefix("b", skip()))
        assert isinstance(p, Parallel)
        assert p.sync_set == frozenset()

    def test_sync_parallel_creation(self):
        """Test creating synchronized parallel."""
        p = parallel(prefix("a", skip()), prefix("a", skip()), {"a"})
        assert isinstance(p, Parallel)
        assert Event("a") in p.sync_set

    def test_interleave_initials(self):
        """Test interleaved parallel initials."""
        p = interleave(prefix("a", skip()), prefix("b", skip()))
        assert p.initials() == frozenset({Event("a"), Event("b")})

    def test_sync_initials(self):
        """Test synchronized parallel only allows common events."""
        p = parallel(prefix("a", skip()), prefix("b", skip()), {"a", "b"})
        # Neither a nor b can proceed alone - both are sync events
        # but only one side has each event
        assert p.initials() == frozenset()

    def test_sync_both_ready(self):
        """Test synchronized event when both sides ready."""
        p = parallel(prefix("a", skip()), prefix("a", skip()), {"a"})
        assert Event("a") in p.initials()

    def test_interleave_after(self):
        """Test interleaved parallel after event."""
        p = interleave(prefix("a", skip()), prefix("b", skip()))
        result = p.after(Event("a"))
        # Left side advances, right stays
        assert isinstance(result, Parallel)
        assert isinstance(result.left, Skip)

    def test_sync_after(self):
        """Test synchronized parallel after event."""
        p = parallel(prefix("a", skip()), prefix("a", skip()), {"a"})
        result = p.after(Event("a"))
        # Both sides advance
        assert isinstance(result, Parallel)
        assert isinstance(result.left, Skip)
        assert isinstance(result.right, Skip)


class TestSequential:
    """Tests for Sequential composition."""

    def test_sequential_creation(self):
        """Test creating sequential composition."""
        p = sequential(prefix("a", skip()), prefix("b", skip()))
        assert isinstance(p, Sequential)

    def test_sequential_initials(self):
        """Test sequential initials from first process."""
        p = sequential(prefix("a", skip()), prefix("b", skip()))
        assert p.initials() == frozenset({Event("a")})

    def test_sequential_after_first(self):
        """Test sequential after first event."""
        p = sequential(prefix("a", skip()), prefix("b", skip()))
        result = p.after(Event("a"))
        # First becomes SKIP, now second's initials available
        assert Event("b") in result.initials()


class TestHiding:
    """Tests for Hiding."""

    def test_hiding_creation(self):
        """Test creating hiding."""
        p = hide(prefix("a", skip()), {"a"})
        assert isinstance(p, Hiding)

    def test_hiding_alphabet(self):
        """Test hidden events removed from alphabet."""
        p = hide(prefix("a", prefix("b", skip())), {"a"})
        assert Event("a") not in p.alphabet()
        assert Event("b") in p.alphabet()

    def test_hiding_initials_tau(self):
        """Test hidden events become τ."""
        p = hide(prefix("a", skip()), {"a"})
        assert TAU in p.initials()
        assert Event("a") not in p.initials()


class TestRecursion:
    """Tests for Recursion."""

    def test_recursion_creation(self):
        """Test creating recursive process."""
        p = rec("X", prefix("a", var("X")))
        assert isinstance(p, Recursion)
        assert p.variable == "X"

    def test_recursion_unfold(self):
        """Test unfolding recursion."""
        p = rec("X", prefix("a", var("X")))
        unfolded = p.unfold()
        # After unfold, should be Prefix with Recursion as continuation
        assert isinstance(unfolded, Prefix)

    def test_recursion_initials(self):
        """Test recursive process initials."""
        p = rec("X", prefix("a", var("X")))
        # Should unfold to get initials
        assert Event("a") in p.initials()


class TestSubstitution:
    """Tests for variable substitution."""

    def test_substitute_variable(self):
        """Test substituting a variable."""
        v = var("X")
        result = substitute(v, "X", skip())
        assert isinstance(result, Skip)

    def test_substitute_prefix(self):
        """Test substituting in prefix."""
        p = prefix("a", var("X"))
        result = substitute(p, "X", skip())
        assert isinstance(result, Prefix)
        assert isinstance(result.continuation, Skip)

    def test_substitute_choice(self):
        """Test substituting in choice."""
        c = choice(var("X"), prefix("a", skip()))
        result = substitute(c, "X", stop())
        assert isinstance(result.left, Stop)

    def test_substitute_shadowed(self):
        """Test shadowed variable not substituted."""
        # μX.(Y) where we substitute Y but X is shadowed
        inner = rec("X", var("Y"))
        result = substitute(inner, "Y", skip())
        assert isinstance(result, Recursion)
