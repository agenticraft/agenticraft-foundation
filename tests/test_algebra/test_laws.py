"""Algebraic law tests for CSP operator correctness."""

from __future__ import annotations

from agenticraft_foundation.algebra.csp import Event, Prefix, Process, Stop
from agenticraft_foundation.algebra.operators import (
    Guard,
    Interrupt,
    Pipe,
    Rename,
    Timeout,
)


class TestInterruptLaws:
    def test_interrupt_by_stop_handler_equals_primary(self):
        """P triangle Stop ~ P (handler with no initials never fires)."""
        a = Event("a")
        p = Prefix(a, Stop())
        interrupted = Interrupt(p, Stop())
        # Stop has no initials, so handler never fires
        assert interrupted.initials() == p.initials()

    def test_interrupt_not_commutative(self):
        """P triangle Q != Q triangle P in general."""
        a, b = Event("a"), Event("b")
        pq = Interrupt(Prefix(a, Stop()), Prefix(b, Stop()))
        qp = Interrupt(Prefix(b, Stop()), Prefix(a, Stop()))
        # After 'a': pq continues primary, qp fires handler
        pq_after = pq.after(a)
        qp_after = qp.after(a)
        assert not isinstance(pq_after, type(qp_after))

    def test_interrupt_preserves_process_type(self):
        """Interrupt(P, Q) is still a Process."""
        p = Interrupt(Stop(), Stop())
        assert isinstance(p, Process)


class TestTimeoutLaws:
    def test_timeout_preserves_process_type(self):
        """Timeout(P, d, Q) is still a Process."""
        p = Timeout(Stop(), 1.0, Stop())
        assert isinstance(p, Process)

    def test_timeout_with_same_process_and_fallback(self):
        """Timeout(P, d, P) should still be valid."""
        a = Event("a")
        p = Prefix(a, Stop())
        t = Timeout(p, 5.0, p)
        assert isinstance(t, Process)
        assert a in t.initials()


class TestGuardLaws:
    def test_true_guard_is_identity(self):
        """Guard(True, P).initials() == P.initials()"""
        a = Event("a")
        p = Prefix(a, Stop())
        guarded = Guard(condition=lambda: True, process=p)
        assert guarded.initials() == p.initials()

    def test_false_guard_is_stop(self):
        """Guard(False, P).initials() == Stop.initials()"""
        a = Event("a")
        p = Prefix(a, Stop())
        guarded = Guard(condition=lambda: False, process=p)
        assert guarded.initials() == Stop().initials()

    def test_guard_preserves_alphabet(self):
        """Guard does not change alphabet regardless of condition."""
        a = Event("a")
        p = Prefix(a, Stop())
        guarded = Guard(condition=lambda: False, process=p)
        assert guarded.alphabet() == p.alphabet()


class TestRenameLaws:
    def test_empty_rename_is_identity(self):
        """Rename(P, {}) has same initials as P."""
        a = Event("a")
        p = Prefix(a, Stop())
        renamed = Rename.from_dict(p, {})
        assert renamed.initials() == p.initials()

    def test_rename_preserves_stop(self):
        """Rename(Stop, f) has no initials (like Stop)."""
        renamed = Rename.from_dict(Stop(), {Event("a"): Event("b")})
        assert renamed.initials() == frozenset()

    def test_rename_preserves_cardinality(self):
        """Rename doesn't add or remove events, just maps them."""
        a, b = Event("a"), Event("b")
        p = Prefix(a, Stop())
        renamed = Rename.from_dict(p, {a: b})
        assert len(renamed.initials()) == len(p.initials())
        assert len(renamed.alphabet()) == len(p.alphabet())


class TestPipeLaws:
    def test_pipe_with_empty_channel_is_independent(self):
        """Pipe(P, Q, empty) -- no sync, processes independent."""
        a, b = Event("a"), Event("b")
        p = Pipe(Prefix(a, Stop()), Prefix(b, Stop()), frozenset())
        assert a in p.alphabet()
        assert b in p.alphabet()

    def test_pipe_preserves_process_type(self):
        """Pipe is a Process."""
        p = Pipe(Stop(), Stop(), frozenset())
        assert isinstance(p, Process)

    def test_channel_events_hidden(self):
        """Channel events do not appear in external alphabet."""
        ch = Event("ch")
        p = Pipe(Prefix(ch, Stop()), Prefix(ch, Stop()), frozenset({ch}))
        assert ch not in p.alphabet()
