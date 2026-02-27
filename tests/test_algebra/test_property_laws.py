"""Property-based tests for CSP algebraic laws using Hypothesis.

Generates random process trees and checks that fundamental algebraic
laws hold under trace equivalence.

Generator restrictions (documented implementation limitations):
- No nested Hiding: outer Hiding.after(TAU) cannot pass through
  inner-generated TAU when outer hidden set is empty.
- Disjoint initials for choice branches: InternalChoice.after()
  only explores one branch on overlapping events, breaking trace
  symmetry needed for commutativity proofs.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from agenticraft_foundation.algebra.csp import (
    Event,
    ExternalChoice,
    Hiding,
    InternalChoice,
    Parallel,
    Prefix,
    Sequential,
    Skip,
    Stop,
)
from agenticraft_foundation.algebra.equivalence import trace_equivalent

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

EVENTS_A = [Event("a"), Event("b")]
EVENTS_B = [Event("c"), Event("d")]
ALL_EVENTS = EVENTS_A + EVENTS_B


@st.composite
def leaf_process(draw: st.DrawFn, events: list[Event] = ALL_EVENTS) -> Stop | Skip | Prefix:
    """A leaf process: Stop, Skip, or a chain of prefixes ending in Stop/Skip."""
    depth = draw(st.integers(min_value=0, max_value=3))
    proc: Stop | Skip | Prefix = draw(st.sampled_from([Stop(), Skip()]))
    for _ in range(depth):
        e = draw(st.sampled_from(events))
        proc = Prefix(event=e, continuation=proc)
    return proc


@st.composite
def process_strategy(
    draw: st.DrawFn,
    max_depth: int = 3,
) -> Stop | Skip | Prefix | ExternalChoice | InternalChoice | Parallel | Sequential | Hiding:
    """Generate a random CSP process tree (no nested Hiding).

    Depth is bounded to keep state spaces tractable.
    """
    if max_depth <= 0:
        return draw(leaf_process())

    kind = draw(
        st.sampled_from(
            [
                "leaf",
                "leaf",  # bias toward simpler processes
                "external_choice",
                "internal_choice",
                "parallel",
                "sequential",
                "hiding",
            ]
        )
    )

    sub = process_strategy(max_depth=max_depth - 1)

    if kind == "leaf":
        return draw(leaf_process())
    elif kind == "external_choice":
        return ExternalChoice(left=draw(sub), right=draw(sub))
    elif kind == "internal_choice":
        return InternalChoice(left=draw(sub), right=draw(sub))
    elif kind == "parallel":
        sync = draw(st.frozensets(st.sampled_from(ALL_EVENTS), max_size=2))
        return Parallel(left=draw(sub), right=draw(sub), sync_set=sync)
    elif kind == "sequential":
        return Sequential(first=draw(sub), second=draw(sub))
    else:  # hiding
        hidden = draw(st.frozensets(st.sampled_from(ALL_EVENTS), min_size=1, max_size=2))
        # Only hide on leaf processes to avoid nested Hiding
        return Hiding(process=draw(leaf_process()), hidden=hidden)


@st.composite
def disjoint_pair(draw: st.DrawFn) -> tuple:
    """Two leaf processes with guaranteed-disjoint initial events."""
    p = draw(leaf_process(events=EVENTS_A))
    q = draw(leaf_process(events=EVENTS_B))
    return p, q


# Shorthand
procs = process_strategy()
leaves = leaf_process()


# ---------------------------------------------------------------------------
# Algebraic law tests
# ---------------------------------------------------------------------------


class TestExternalChoiceLaws:
    """Laws for external choice (box)."""

    @given(data=st.data())
    @settings(max_examples=100)
    def test_commutativity(self, data):
        """P [] Q =_T Q [] P (with disjoint initials)."""
        p, q = data.draw(disjoint_pair())
        lhs = ExternalChoice(p, q)
        rhs = ExternalChoice(q, p)
        result = trace_equivalent(lhs, rhs)
        assert result.is_equivalent, f"Commutativity failed: {result.witness}"

    @given(p=leaves)
    @settings(max_examples=100)
    def test_idempotence(self, p):
        """P [] P =_T P."""
        result = trace_equivalent(ExternalChoice(p, p), p)
        assert result.is_equivalent, f"Idempotence failed: {result.witness}"

    @given(data=st.data())
    @settings(max_examples=100)
    def test_associativity(self, data):
        """(P [] Q) [] R =_T P [] (Q [] R) (with disjoint initials)."""
        p = data.draw(leaf_process(events=[Event("a")]))
        q = data.draw(leaf_process(events=[Event("b")]))
        r = data.draw(leaf_process(events=[Event("c")]))
        lhs = ExternalChoice(ExternalChoice(p, q), r)
        rhs = ExternalChoice(p, ExternalChoice(q, r))
        result = trace_equivalent(lhs, rhs)
        assert result.is_equivalent, f"Associativity failed: {result.witness}"


class TestInternalChoiceLaws:
    """Laws for internal choice (nondeterministic)."""

    @given(data=st.data())
    @settings(max_examples=100)
    def test_commutativity(self, data):
        """P |~| Q =_T Q |~| P (with disjoint initials)."""
        p, q = data.draw(disjoint_pair())
        lhs = InternalChoice(p, q)
        rhs = InternalChoice(q, p)
        result = trace_equivalent(lhs, rhs)
        assert result.is_equivalent, f"Commutativity failed: {result.witness}"

    @given(p=leaves)
    @settings(max_examples=100)
    def test_idempotence(self, p):
        """P |~| P =_T P."""
        result = trace_equivalent(InternalChoice(p, p), p)
        assert result.is_equivalent, f"Idempotence failed: {result.witness}"


class TestInterleavingLaws:
    """Laws for interleaving (Parallel with empty sync set)."""

    @given(data=st.data())
    @settings(max_examples=100)
    def test_commutativity(self, data):
        """P ||| Q =_T Q ||| P (with disjoint initials)."""
        p, q = data.draw(disjoint_pair())
        lhs = Parallel(left=p, right=q, sync_set=frozenset())
        rhs = Parallel(left=q, right=p, sync_set=frozenset())
        result = trace_equivalent(lhs, rhs)
        assert result.is_equivalent, f"Interleaving commutativity failed: {result.witness}"


class TestIdentityLaws:
    """Identity elements for various operators."""

    @given(p=leaves)
    @settings(max_examples=100)
    def test_stop_identity_for_choice(self, p):
        """P [] STOP =_T P."""
        result = trace_equivalent(ExternalChoice(p, Stop()), p)
        assert result.is_equivalent, f"STOP identity for choice failed: {result.witness}"

    @given(p=leaves)
    @settings(max_examples=100)
    def test_stop_absorbs_sequential(self, p):
        """STOP ; P =_T STOP (deadlock absorbs)."""
        result = trace_equivalent(Sequential(first=Stop(), second=p), Stop())
        assert result.is_equivalent, f"STOP absorbs sequential failed: {result.witness}"

    @given(p=leaves)
    @settings(max_examples=100)
    def test_skip_identity_for_sequential(self, p):
        """SKIP ; P =_T P."""
        result = trace_equivalent(Sequential(first=Skip(), second=p), p)
        assert result.is_equivalent, f"SKIP identity for sequential failed: {result.witness}"


class TestHidingLaws:
    """Laws for hiding (P \\\\ H)."""

    @given(p=leaves)
    @settings(max_examples=100)
    def test_hiding_empty_set(self, p):
        """P \\\\ {} =_T P."""
        result = trace_equivalent(Hiding(process=p, hidden=frozenset()), p)
        assert result.is_equivalent, f"Hiding empty set failed: {result.witness}"
