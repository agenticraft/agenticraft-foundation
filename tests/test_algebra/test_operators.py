"""Tests for the 5 extended CSP operators."""

from __future__ import annotations

import pytest

from agenticraft_foundation.algebra.csp import Event, Prefix, ProcessKind, Stop
from agenticraft_foundation.algebra.operators import (
    TIMEOUT_EVENT,
    Guard,
    Interrupt,
    Pipe,
    Rename,
    Timeout,
)


class TestInterrupt:
    def test_kind_is_interrupt(self):
        p = Interrupt(Stop(), Stop())
        assert p.kind == ProcessKind.INTERRUPT

    def test_alphabet_is_union(self):
        a, b = Event("a"), Event("b")
        p = Interrupt(Prefix(a, Stop()), Prefix(b, Stop()))
        assert p.alphabet() == frozenset({a, b})

    def test_initials_include_both(self):
        a, b = Event("a"), Event("b")
        p = Interrupt(Prefix(a, Stop()), Prefix(b, Stop()))
        assert a in p.initials()
        assert b in p.initials()

    def test_after_primary_event_continues_primary(self):
        a, b = Event("a"), Event("b")
        p = Interrupt(Prefix(a, Stop()), Prefix(b, Stop()))
        result = p.after(a)
        assert isinstance(result, Interrupt)

    def test_after_handler_event_switches_to_handler(self):
        a, b = Event("a"), Event("b")
        p = Interrupt(Prefix(a, Stop()), Prefix(b, Stop()))
        result = p.after(b)
        # After interrupt fires, we're in handler's continuation
        assert not isinstance(result, Interrupt)

    def test_handler_priority_over_primary(self):
        """When event is in both primary and handler initials, handler wins."""
        a = Event("a")
        p = Interrupt(
            Prefix(a, Prefix(Event("p_next"), Stop())),
            Prefix(a, Prefix(Event("h_next"), Stop())),
        )
        result = p.after(a)
        # Handler should take priority
        assert not isinstance(result, Interrupt)
        assert Event("h_next") in result.initials()

    def test_after_invalid_event_raises(self):
        a, b, c = Event("a"), Event("b"), Event("c")
        p = Interrupt(Prefix(a, Stop()), Prefix(b, Stop()))
        with pytest.raises(ValueError, match="not in alphabet"):
            p.after(c)

    def test_repr_uses_triangle(self):
        p = Interrupt(Stop(), Stop())
        assert "\u25b3" in repr(p)

    def test_frozen_and_hashable(self):
        p = Interrupt(Stop(), Stop())
        hash(p)  # should not raise


class TestTimeout:
    def test_kind_is_timeout(self):
        p = Timeout(Stop(), 10.0, Stop())
        assert p.kind == ProcessKind.TIMEOUT

    def test_positive_duration_required(self):
        with pytest.raises(ValueError, match="positive"):
            Timeout(Stop(), -1.0, Stop())
        with pytest.raises(ValueError, match="positive"):
            Timeout(Stop(), 0.0, Stop())

    def test_alphabet_includes_timeout_event(self):
        p = Timeout(Prefix(Event("a"), Stop()), 5.0, Stop())
        assert TIMEOUT_EVENT in p.alphabet()

    def test_initials_include_timeout_event(self):
        p = Timeout(Prefix(Event("a"), Stop()), 5.0, Stop())
        assert TIMEOUT_EVENT in p.initials()
        assert Event("a") in p.initials()

    def test_after_timeout_event_gives_fallback(self):
        fallback = Prefix(Event("cached"), Stop())
        p = Timeout(Prefix(Event("a"), Stop()), 5.0, fallback)
        result = p.after(TIMEOUT_EVENT)
        assert result is fallback

    def test_after_process_event_continues_with_timeout(self):
        a = Event("a")
        p = Timeout(Prefix(a, Stop()), 5.0, Stop())
        result = p.after(a)
        assert isinstance(result, Timeout)

    def test_after_invalid_event_raises(self):
        p = Timeout(Prefix(Event("a"), Stop()), 5.0, Stop())
        with pytest.raises(ValueError, match="not in alphabet"):
            p.after(Event("z"))

    def test_repr(self):
        p = Timeout(Stop(), 1.0, Stop())
        assert "Timeout" in repr(p)

    def test_frozen_and_hashable(self):
        p = Timeout(Stop(), 1.0, Stop())
        hash(p)  # should not raise


class TestGuard:
    def test_kind_is_guard(self):
        p = Guard(condition=lambda: True, process=Stop())
        assert p.kind == ProcessKind.GUARD

    def test_true_guard_exposes_process_initials(self):
        a = Event("a")
        p = Guard(condition=lambda: True, process=Prefix(a, Stop()))
        assert a in p.initials()

    def test_false_guard_has_empty_initials(self):
        a = Event("a")
        p = Guard(condition=lambda: False, process=Prefix(a, Stop()))
        assert p.initials() == frozenset()

    def test_alphabet_always_includes_process_alphabet(self):
        a = Event("a")
        p = Guard(condition=lambda: False, process=Prefix(a, Stop()))
        assert a in p.alphabet()

    def test_after_true_guard_proceeds(self):
        a = Event("a")
        p = Guard(condition=lambda: True, process=Prefix(a, Stop()))
        result = p.after(a)
        assert isinstance(result, Stop)

    def test_after_false_guard_raises(self):
        a = Event("a")
        p = Guard(condition=lambda: False, process=Prefix(a, Stop()))
        with pytest.raises(ValueError, match="not in initials"):
            p.after(a)

    def test_mutable_condition(self):
        """Guard re-evaluates condition on each call."""
        state = {"active": False}
        a = Event("a")
        p = Guard(condition=lambda: state["active"], process=Prefix(a, Stop()))
        assert p.initials() == frozenset()
        state["active"] = True
        assert a in p.initials()

    def test_guard_equality_same_condition(self):
        cond = lambda: True  # noqa: E731
        p1 = Guard(condition=cond, process=Stop())
        p2 = Guard(condition=cond, process=Stop())
        assert p1 == p2

    def test_guard_inequality_different_conditions(self):
        p1 = Guard(condition=lambda: True, process=Stop())
        p2 = Guard(condition=lambda: True, process=Stop())
        # Different lambda objects -> not equal
        assert p1 != p2

    def test_repr(self):
        p = Guard(condition=lambda: True, process=Stop())
        assert "Guard" in repr(p)
        assert "<condition>" in repr(p)


class TestRename:
    def test_kind_is_rename(self):
        p = Rename.from_dict(Stop(), {})
        assert p.kind == ProcessKind.RENAME

    def test_renamed_event_appears_in_initials(self):
        a, b = Event("a"), Event("b")
        p = Rename.from_dict(Prefix(a, Stop()), {a: b})
        assert b in p.initials()
        assert a not in p.initials()

    def test_alphabet_is_transformed(self):
        a, b = Event("a"), Event("b")
        p = Rename.from_dict(Prefix(a, Stop()), {a: b})
        assert b in p.alphabet()
        assert a not in p.alphabet()

    def test_after_renamed_event(self):
        a, b = Event("a"), Event("b")
        p = Rename.from_dict(Prefix(a, Stop()), {a: b})
        result = p.after(b)
        assert isinstance(result, Rename)

    def test_unmapped_events_unchanged(self):
        a, b, c = Event("a"), Event("b"), Event("c")
        inner = Prefix(a, Prefix(c, Stop()))
        p = Rename.from_dict(inner, {a: b})
        assert b in p.initials()  # a -> b
        # c is unmapped, stays as c in alphabet
        assert c in p.alphabet()

    def test_after_invalid_event_raises(self):
        a, b = Event("a"), Event("b")
        p = Rename.from_dict(Prefix(a, Stop()), {a: b})
        with pytest.raises(ValueError, match="not in initials"):
            p.after(Event("z"))

    def test_repr(self):
        a, b = Event("a"), Event("b")
        p = Rename.from_dict(Prefix(a, Stop()), {a: b})
        assert "[[" in repr(p)

    def test_frozen_and_hashable(self):
        p = Rename.from_dict(Stop(), {})
        hash(p)  # should not raise


class TestPipe:
    def test_kind_is_pipe(self):
        p = Pipe(Stop(), Stop(), frozenset())
        assert p.kind == ProcessKind.PIPE

    def test_channel_hidden_from_alphabet(self):
        a, ch, b = Event("a"), Event("channel"), Event("b")
        p = Pipe(
            producer=Prefix(a, Prefix(ch, Stop())),
            consumer=Prefix(ch, Prefix(b, Stop())),
            channel=frozenset({ch}),
        )
        assert a in p.alphabet()
        assert b in p.alphabet()
        assert ch not in p.alphabet()

    def test_after_channel_event_raises(self):
        ch = Event("channel")
        p = Pipe(Prefix(ch, Stop()), Prefix(ch, Stop()), frozenset({ch}))
        with pytest.raises(ValueError, match="hidden"):
            p.after(ch)

    def test_after_producer_event(self):
        a, ch = Event("a"), Event("ch")
        p = Pipe(
            producer=Prefix(a, Prefix(ch, Stop())),
            consumer=Prefix(ch, Stop()),
            channel=frozenset({ch}),
        )
        result = p.after(a)
        assert isinstance(result, Pipe)

    def test_empty_channel_means_independent(self):
        a, b = Event("a"), Event("b")
        p = Pipe(Prefix(a, Stop()), Prefix(b, Stop()), frozenset())
        assert a in p.alphabet()
        assert b in p.alphabet()

    def test_repr(self):
        p = Pipe(Stop(), Stop(), frozenset({Event("ch")}))
        assert "|>" in repr(p)

    def test_frozen_and_hashable(self):
        p = Pipe(Stop(), Stop(), frozenset())
        hash(p)  # should not raise
