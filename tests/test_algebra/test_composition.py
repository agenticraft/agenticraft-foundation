"""Composition tests: new operators work with existing operators and analysis."""

from __future__ import annotations

from agenticraft_foundation.algebra.csp import (
    Event,
    ExternalChoice,
    Parallel,
    Prefix,
    Recursion,
    Sequential,
    Stop,
    Variable,
    substitute,
)
from agenticraft_foundation.algebra.operators import (
    TIMEOUT_EVENT,
    Guard,
    Interrupt,
    Pipe,
    Rename,
    Timeout,
)
from agenticraft_foundation.algebra.semantics import (
    build_lts,
    detect_deadlock,
    is_deadlock_free,
    traces,
)


class TestOperatorComposition:
    def test_interrupt_inside_parallel(self):
        """Parallel(P triangle Q, R) type-checks and produces valid Process."""
        a, b, c = Event("a"), Event("b"), Event("c")
        interruptible = Interrupt(Prefix(a, Stop()), Prefix(b, Stop()))
        other = Prefix(c, Stop())
        composed = Parallel(left=interruptible, right=other, sync_set=frozenset())
        assert a in composed.alphabet()
        assert b in composed.alphabet()
        assert c in composed.alphabet()

    def test_timeout_in_sequential(self):
        """Sequential(Timeout(P, d, Q), R) produces valid Process."""
        a, b = Event("a"), Event("b")
        bounded = Timeout(Prefix(a, Stop()), 5.0, Stop())
        after_timeout = Prefix(b, Stop())
        composed = Sequential(first=bounded, second=after_timeout)
        assert isinstance(composed, Sequential)
        assert a in composed.alphabet()

    def test_guard_with_external_choice(self):
        """ExternalChoice(Guard(b, P), Guard(c, Q)) works."""
        a, b = Event("a"), Event("b")
        left = Guard(condition=lambda: True, process=Prefix(a, Stop()))
        right = Guard(condition=lambda: True, process=Prefix(b, Stop()))
        composed = ExternalChoice(left=left, right=right)
        assert a in composed.initials()
        assert b in composed.initials()

    def test_renamed_interrupt(self):
        """Rename(Interrupt(P, Q), mapping) works."""
        a, b, c = Event("a"), Event("b"), Event("c")
        inner = Interrupt(Prefix(a, Stop()), Prefix(b, Stop()))
        renamed = Rename.from_dict(inner, {a: c})
        assert c in renamed.initials()
        assert a not in renamed.initials()
        assert b in renamed.initials()

    def test_nested_interrupt_timeout(self):
        """Timeout(Interrupt(P, Q), d, R) -- interrupt inside timeout."""
        a, b, c = Event("a"), Event("b"), Event("c")
        inner = Interrupt(Prefix(a, Stop()), Prefix(b, Stop()))
        fallback = Prefix(c, Stop())
        composed = Timeout(inner, 10.0, fallback)
        assert a in composed.initials()
        assert b in composed.initials()
        assert TIMEOUT_EVENT in composed.initials()

    def test_all_operators_work_with_build_lts(self):
        """build_lts() works on all new operators."""
        a, b = Event("a"), Event("b")
        # Interrupt
        lts = build_lts(Interrupt(Prefix(a, Stop()), Prefix(b, Stop())))
        assert len(lts.states) > 0
        # Timeout
        lts = build_lts(Timeout(Prefix(a, Stop()), 5.0, Stop()))
        assert len(lts.states) > 0
        # Guard
        lts = build_lts(Guard(condition=lambda: True, process=Prefix(a, Stop())))
        assert len(lts.states) > 0
        # Rename
        lts = build_lts(Rename.from_dict(Prefix(a, Stop()), {a: b}))
        assert len(lts.states) > 0
        # Pipe
        ch = Event("ch")
        lts = build_lts(
            Pipe(Prefix(a, Prefix(ch, Stop())), Prefix(ch, Prefix(b, Stop())), frozenset({ch}))
        )
        assert len(lts.states) > 0

    def test_all_operators_work_with_traces(self):
        """traces() from semantics.py works on new operators."""
        a, b = Event("a"), Event("b")
        proc = Interrupt(Prefix(a, Stop()), Prefix(b, Stop()))
        lts = build_lts(proc)
        t = list(traces(lts, max_length=10))
        assert len(t) > 0
        # Should include traces for both primary and handler paths
        assert (a,) in t or (b,) in t

    def test_all_operators_work_with_detect_deadlock(self):
        """detect_deadlock() from semantics.py works on new operators."""
        a = Event("a")
        proc = Timeout(Prefix(a, Stop()), 5.0, Stop())
        lts = build_lts(proc)
        dl = detect_deadlock(lts)
        # Stop is a deadlock state
        assert dl.has_deadlock is True

    def test_is_deadlock_free_on_new_operators(self):
        """is_deadlock_free() works on new operators."""
        a = Event("a")
        proc = Interrupt(Prefix(a, Stop()), Stop())
        result = is_deadlock_free(proc)
        # Stop() is deadlock, so the process has deadlock
        assert result is False


class TestSubstituteWithNewOperators:
    def test_substitute_interrupt(self):
        """substitute() correctly handles Interrupt."""
        a, b = Event("a"), Event("b")
        body = Interrupt(
            primary=Prefix(a, Variable("X")),
            handler=Prefix(b, Stop()),
        )
        replacement = Prefix(Event("c"), Stop())
        result = substitute(body, "X", replacement)
        assert isinstance(result, Interrupt)

    def test_substitute_timeout(self):
        """substitute() correctly handles Timeout."""
        a = Event("a")
        body = Timeout(
            process=Prefix(a, Variable("X")),
            duration=5.0,
            fallback=Variable("X"),
        )
        replacement = Stop()
        result = substitute(body, "X", replacement)
        assert isinstance(result, Timeout)

    def test_substitute_guard(self):
        """substitute() correctly handles Guard."""
        a = Event("a")
        body = Guard(condition=lambda: True, process=Variable("X"))
        replacement = Prefix(a, Stop())
        result = substitute(body, "X", replacement)
        assert isinstance(result, Guard)

    def test_substitute_rename(self):
        """substitute() correctly handles Rename."""
        body = Rename.from_dict(Variable("X"), {})
        replacement = Stop()
        result = substitute(body, "X", replacement)
        assert isinstance(result, Rename)

    def test_substitute_pipe(self):
        """substitute() correctly handles Pipe."""
        body = Pipe(
            producer=Variable("X"),
            consumer=Variable("X"),
            channel=frozenset(),
        )
        replacement = Stop()
        result = substitute(body, "X", replacement)
        assert isinstance(result, Pipe)

    def test_recursion_with_interrupt(self):
        """rec('X', Interrupt(a -> X, b -> Stop)) correctly unfolds."""
        a, b = Event("a"), Event("b")
        body = Interrupt(
            primary=Prefix(a, Variable("X")),
            handler=Prefix(b, Stop()),
        )
        rec_proc = Recursion(variable="X", body=body)
        unfolded = rec_proc.unfold()
        assert isinstance(unfolded, Interrupt)
        # Primary should have 'a' in its initials
        assert a in unfolded.initials()

    def test_recursion_with_timeout(self):
        """rec('X', Timeout(a -> X, d, Stop)) correctly unfolds."""
        a = Event("a")
        body = Timeout(
            process=Prefix(a, Variable("X")),
            duration=10.0,
            fallback=Stop(),
        )
        rec_proc = Recursion(variable="X", body=body)
        unfolded = rec_proc.unfold()
        assert isinstance(unfolded, Timeout)

    def test_substitute_preserves_wrong_variable(self):
        """substitute() with non-matching variable returns unchanged."""
        body = Interrupt(primary=Variable("Y"), handler=Stop())
        result = substitute(body, "X", Stop())
        assert isinstance(result, Interrupt)
        # Variable Y should remain
        assert isinstance(result.primary, Variable)
        assert result.primary.name == "Y"
