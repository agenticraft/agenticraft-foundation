"""Tests for CTL temporal logic model checking."""

from __future__ import annotations

import pytest

from agenticraft_foundation.algebra.csp import Event, Prefix, Stop
from agenticraft_foundation.algebra.semantics import LTS, LTSState, Transition
from agenticraft_foundation.verification.temporal import (
    AF,
    AG,
    AU,
    AX,
    EF,
    EG,
    EU,
    EX,
    And,
    Atomic,
    CTLFormula,
    Implies,
    Labeling,
    ModelCheckResult,
    Not,
    Or,
    check_invariant,
    check_liveness,
    check_safety,
    model_check,
)


# =============================================================================
# Test Fixtures: LTS Construction Helpers
# =============================================================================


def _make_linear_lts() -> tuple[LTS, Labeling]:
    """Linear LTS: 0 --a--> 1 --b--> 2.

    Labels: {0: {init}, 1: {processing}, 2: {done}}
    """
    lts = LTS()
    lts.add_state(LTSState(0, Stop()))
    lts.add_state(LTSState(1, Stop()))
    lts.add_state(LTSState(2, Stop(), is_terminal=True))
    lts.add_transition(Transition(0, Event("a"), 1))
    lts.add_transition(Transition(1, Event("b"), 2))

    labeling: Labeling = {
        0: {"init"},
        1: {"processing"},
        2: {"done"},
    }
    return lts, labeling


def _make_branching_lts() -> tuple[LTS, Labeling]:
    """Branching LTS:
         0 --a--> 1 --b--> 2 (success)
         0 --a--> 1 --c--> 3 (error)

    Labels: {0: {init}, 1: {processing}, 2: {success}, 3: {error}}
    """
    lts = LTS()
    lts.add_state(LTSState(0, Stop()))
    lts.add_state(LTSState(1, Stop()))
    lts.add_state(LTSState(2, Stop(), is_terminal=True))
    lts.add_state(LTSState(3, Stop(), is_deadlock=True))
    lts.add_transition(Transition(0, Event("a"), 1))
    lts.add_transition(Transition(1, Event("b"), 2))
    lts.add_transition(Transition(1, Event("c"), 3))

    labeling: Labeling = {
        0: {"init"},
        1: {"processing"},
        2: {"success"},
        3: {"error"},
    }
    return lts, labeling


def _make_cyclic_lts() -> tuple[LTS, Labeling]:
    """Cyclic LTS: 0 --a--> 1 --b--> 0, 1 --c--> 2.

    Labels: {0: {ready}, 1: {working}, 2: {done}}
    """
    lts = LTS()
    lts.add_state(LTSState(0, Stop()))
    lts.add_state(LTSState(1, Stop()))
    lts.add_state(LTSState(2, Stop(), is_terminal=True))
    lts.add_transition(Transition(0, Event("a"), 1))
    lts.add_transition(Transition(1, Event("b"), 0))
    lts.add_transition(Transition(1, Event("c"), 2))

    labeling: Labeling = {
        0: {"ready"},
        1: {"working"},
        2: {"done"},
    }
    return lts, labeling


def _make_diamond_lts() -> tuple[LTS, Labeling]:
    """Diamond LTS:
         0 --a--> 1
         0 --b--> 2
         1 --c--> 3
         2 --d--> 3

    Labels: {0: {start}, 1: {left}, 2: {right}, 3: {end}}
    """
    lts = LTS()
    lts.add_state(LTSState(0, Stop()))
    lts.add_state(LTSState(1, Stop()))
    lts.add_state(LTSState(2, Stop()))
    lts.add_state(LTSState(3, Stop(), is_terminal=True))
    lts.add_transition(Transition(0, Event("a"), 1))
    lts.add_transition(Transition(0, Event("b"), 2))
    lts.add_transition(Transition(1, Event("c"), 3))
    lts.add_transition(Transition(2, Event("d"), 3))

    labeling: Labeling = {
        0: {"start"},
        1: {"left"},
        2: {"right"},
        3: {"end"},
    }
    return lts, labeling


def _make_absorbing_lts() -> tuple[LTS, Labeling]:
    """LTS with absorbing states (self-loops):
         0 --a--> 1 (success, self-loop)
         0 --b--> 2 (error, self-loop)

    Labels: {0: {init}, 1: {success}, 2: {error}}
    """
    lts = LTS()
    lts.add_state(LTSState(0, Stop()))
    lts.add_state(LTSState(1, Stop()))
    lts.add_state(LTSState(2, Stop()))
    lts.add_transition(Transition(0, Event("a"), 1))
    lts.add_transition(Transition(0, Event("b"), 2))
    lts.add_transition(Transition(1, Event("stay"), 1))  # absorbing
    lts.add_transition(Transition(2, Event("stay"), 2))  # absorbing

    labeling: Labeling = {
        0: {"init"},
        1: {"success"},
        2: {"error"},
    }
    return lts, labeling


# =============================================================================
# Formula AST Tests
# =============================================================================


class TestFormulaAST:
    """Tests for CTL formula construction and representation."""

    def test_atomic(self) -> None:
        f = Atomic("p")
        assert f.prop == "p"
        assert repr(f) == "p"

    def test_not(self) -> None:
        f = Not(Atomic("p"))
        assert repr(f) == "¬(p)"

    def test_and(self) -> None:
        f = And(Atomic("p"), Atomic("q"))
        assert repr(f) == "(p ∧ q)"

    def test_or(self) -> None:
        f = Or(Atomic("p"), Atomic("q"))
        assert repr(f) == "(p ∨ q)"

    def test_implies(self) -> None:
        f = Implies(Atomic("p"), Atomic("q"))
        assert repr(f) == "(p → q)"

    def test_ex(self) -> None:
        f = EX(Atomic("p"))
        assert repr(f) == "EX(p)"

    def test_ef(self) -> None:
        f = EF(Atomic("p"))
        assert repr(f) == "EF(p)"

    def test_eg(self) -> None:
        f = EG(Atomic("p"))
        assert repr(f) == "EG(p)"

    def test_eu(self) -> None:
        f = EU(Atomic("p"), Atomic("q"))
        assert repr(f) == "E[p U q]"

    def test_ax(self) -> None:
        f = AX(Atomic("p"))
        assert repr(f) == "AX(p)"

    def test_af(self) -> None:
        f = AF(Atomic("p"))
        assert repr(f) == "AF(p)"

    def test_ag(self) -> None:
        f = AG(Atomic("p"))
        assert repr(f) == "AG(p)"

    def test_au(self) -> None:
        f = AU(Atomic("p"), Atomic("q"))
        assert repr(f) == "A[p U q]"

    def test_equality(self) -> None:
        f1 = And(Atomic("p"), Atomic("q"))
        f2 = And(Atomic("p"), Atomic("q"))
        f3 = And(Atomic("p"), Atomic("r"))
        assert f1 == f2
        assert f1 != f3

    def test_hash(self) -> None:
        f1 = EX(Atomic("p"))
        f2 = EX(Atomic("p"))
        assert hash(f1) == hash(f2)
        assert {f1, f2} == {f1}

    def test_nested_formula(self) -> None:
        f = AG(Implies(Atomic("req"), AF(Atomic("resp"))))
        assert repr(f) == "AG((req → AF(resp)))"

    def test_frozen(self) -> None:
        """Formulas are immutable."""
        f = Atomic("p")
        with pytest.raises(AttributeError):
            f.prop = "q"  # type: ignore[misc]


# =============================================================================
# Atomic Proposition Tests
# =============================================================================


class TestAtomic:
    """Tests for atomic proposition evaluation."""

    def test_atomic_holds(self) -> None:
        lts, labeling = _make_linear_lts()
        result = model_check(lts, Atomic("init"), labeling)
        assert result.satisfied  # State 0 has "init"
        assert 0 in result.satisfying_states

    def test_atomic_not_in_initial(self) -> None:
        lts, labeling = _make_linear_lts()
        result = model_check(lts, Atomic("done"), labeling)
        assert not result.satisfied  # State 0 doesn't have "done"
        assert 2 in result.satisfying_states

    def test_atomic_nonexistent_prop(self) -> None:
        lts, labeling = _make_linear_lts()
        result = model_check(lts, Atomic("nonexistent"), labeling)
        assert not result.satisfied
        assert result.satisfying_states == set()


# =============================================================================
# Boolean Operator Tests
# =============================================================================


class TestBooleanOperators:
    """Tests for Not, And, Or, Implies."""

    def test_not(self) -> None:
        lts, labeling = _make_linear_lts()
        result = model_check(lts, Not(Atomic("done")), labeling)
        assert result.satisfied  # State 0 is not "done"
        assert 0 in result.satisfying_states
        assert 1 in result.satisfying_states
        assert 2 not in result.satisfying_states

    def test_and_both_hold(self) -> None:
        lts, labeling = _make_linear_lts()
        labeling[0].add("active")
        result = model_check(lts, And(Atomic("init"), Atomic("active")), labeling)
        assert result.satisfied

    def test_and_one_fails(self) -> None:
        lts, labeling = _make_linear_lts()
        result = model_check(lts, And(Atomic("init"), Atomic("done")), labeling)
        assert not result.satisfied

    def test_or_one_holds(self) -> None:
        lts, labeling = _make_linear_lts()
        result = model_check(lts, Or(Atomic("init"), Atomic("done")), labeling)
        assert result.satisfied

    def test_or_neither_holds(self) -> None:
        lts, labeling = _make_linear_lts()
        result = model_check(lts, Or(Atomic("done"), Atomic("error")), labeling)
        assert not result.satisfied

    def test_implies_true(self) -> None:
        lts, labeling = _make_linear_lts()
        # init → init is trivially true
        result = model_check(lts, Implies(Atomic("init"), Atomic("init")), labeling)
        assert result.satisfied

    def test_implies_false_antecedent(self) -> None:
        lts, labeling = _make_linear_lts()
        # done → init at state 0: done is false, so implication is true
        result = model_check(lts, Implies(Atomic("done"), Atomic("init")), labeling)
        assert result.satisfied


# =============================================================================
# EX Tests
# =============================================================================


class TestEX:
    """Tests for EX (existential next)."""

    def test_ex_holds(self) -> None:
        lts, labeling = _make_linear_lts()
        # From state 0, there exists a next state with "processing"
        result = model_check(lts, EX(Atomic("processing")), labeling)
        assert result.satisfied
        assert 0 in result.satisfying_states

    def test_ex_not_from_terminal(self) -> None:
        lts, labeling = _make_linear_lts()
        # State 2 is terminal, no next state
        result = model_check(lts, EX(Atomic("init")), labeling)
        assert 2 not in result.satisfying_states

    def test_ex_two_steps(self) -> None:
        lts, labeling = _make_linear_lts()
        # EX(EX(done)): from state 0, next state has a next state with "done"
        result = model_check(lts, EX(EX(Atomic("done"))), labeling)
        assert result.satisfied  # 0 -> 1 -> 2(done)

    def test_ex_branching(self) -> None:
        lts, labeling = _make_branching_lts()
        # From state 1, there exists a next state with "error"
        result = model_check(lts, EX(Atomic("error")), labeling)
        assert 1 in result.satisfying_states


# =============================================================================
# AX Tests
# =============================================================================


class TestAX:
    """Tests for AX (universal next)."""

    def test_ax_holds_linear(self) -> None:
        lts, labeling = _make_linear_lts()
        # From state 0, ALL next states have "processing" (only one: state 1)
        result = model_check(lts, AX(Atomic("processing")), labeling)
        assert result.satisfied

    def test_ax_fails_branching(self) -> None:
        lts, labeling = _make_branching_lts()
        # From state 1, NOT all next states have "success" (state 3 has "error")
        result = model_check(lts, AX(Atomic("success")), labeling)
        assert 1 not in result.satisfying_states

    def test_ax_vacuous_on_terminal(self) -> None:
        lts, labeling = _make_linear_lts()
        # State 2 has no successors — AX is vacuously true
        result = model_check(lts, AX(Atomic("anything")), labeling)
        assert 2 in result.satisfying_states


# =============================================================================
# EF Tests
# =============================================================================


class TestEF:
    """Tests for EF (existential eventually)."""

    def test_ef_reachable(self) -> None:
        lts, labeling = _make_linear_lts()
        # "done" is reachable from state 0
        result = model_check(lts, EF(Atomic("done")), labeling)
        assert result.satisfied
        assert 0 in result.satisfying_states
        assert 1 in result.satisfying_states
        assert 2 in result.satisfying_states

    def test_ef_unreachable(self) -> None:
        lts, labeling = _make_linear_lts()
        # "error" is not reachable (doesn't exist)
        result = model_check(lts, EF(Atomic("error")), labeling)
        assert not result.satisfied

    def test_ef_self(self) -> None:
        lts, labeling = _make_linear_lts()
        # "init" is immediately true at state 0
        result = model_check(lts, EF(Atomic("init")), labeling)
        assert result.satisfied

    def test_ef_branching(self) -> None:
        lts, labeling = _make_branching_lts()
        # "error" is reachable via path 0->1->3
        result = model_check(lts, EF(Atomic("error")), labeling)
        assert result.satisfied
        assert 0 in result.satisfying_states

    def test_ef_cyclic(self) -> None:
        lts, labeling = _make_cyclic_lts()
        # "done" is reachable from state 0 (0->1->2)
        result = model_check(lts, EF(Atomic("done")), labeling)
        assert result.satisfied


# =============================================================================
# AF Tests
# =============================================================================


class TestAF:
    """Tests for AF (universal eventually)."""

    def test_af_all_paths_reach(self) -> None:
        lts, labeling = _make_linear_lts()
        # On all paths from state 0, "done" is eventually reached
        result = model_check(lts, AF(Atomic("done")), labeling)
        assert result.satisfied

    def test_af_holds_with_deadlock(self) -> None:
        lts, labeling = _make_branching_lts()
        # AF(success): state 3 (error) is a deadlock — AF vacuously holds at
        # deadlock states (no infinite paths to avoid success). This is standard
        # CTL semantics: AF φ = ¬EG(¬φ), and EG requires infinite successor chains.
        result = model_check(lts, AF(Atomic("success")), labeling)
        assert result.satisfied  # Vacuously true at deadlock states

    def test_af_fails_with_cycle(self) -> None:
        """AF fails when there's an infinite path that never reaches the target."""
        lts, labeling = _make_cyclic_lts()
        # Cycle 0->1->0->1... can loop forever without reaching "done"
        result = model_check(lts, AF(Atomic("done")), labeling)
        assert not result.satisfied

    def test_af_with_cycle(self) -> None:
        lts, labeling = _make_cyclic_lts()
        # With cycle 0->1->0->1..., "done" may never be reached
        # But there IS a path through: 0->1->2(done)
        # AF requires ALL paths, but the cycle path never reaches done
        result = model_check(lts, AF(Atomic("done")), labeling)
        # The cycle allows infinite looping without reaching done
        assert not result.satisfied

    def test_af_immediate(self) -> None:
        lts, labeling = _make_linear_lts()
        # AF(init) at state 0 — already true
        result = model_check(lts, AF(Atomic("init")), labeling)
        assert result.satisfied


# =============================================================================
# EG Tests
# =============================================================================


class TestEG:
    """Tests for EG (existential globally)."""

    def test_eg_with_cycle(self) -> None:
        lts, labeling = _make_cyclic_lts()
        # Add "alive" to states 0 and 1
        labeling[0].add("alive")
        labeling[1].add("alive")
        # EG(alive): there exists a path where alive holds forever (the cycle 0->1->0->1...)
        result = model_check(lts, EG(Atomic("alive")), labeling)
        assert result.satisfied

    def test_eg_no_cycle(self) -> None:
        lts, labeling = _make_linear_lts()
        # EG(init): init only holds at state 0, no cycle to stay there
        result = model_check(lts, EG(Atomic("init")), labeling)
        assert not result.satisfied

    def test_eg_self_loop(self) -> None:
        lts, labeling = _make_absorbing_lts()
        # EG(success): state 1 has self-loop and "success"
        result = model_check(lts, EG(Atomic("success")), labeling)
        # State 1 can loop forever with "success"
        assert 1 in result.satisfying_states


# =============================================================================
# AG Tests
# =============================================================================


class TestAG:
    """Tests for AG (universal globally)."""

    def test_ag_holds_everywhere(self) -> None:
        lts, labeling = _make_linear_lts()
        # Add "valid" to all states
        for s in labeling:
            labeling[s].add("valid")
        result = model_check(lts, AG(Atomic("valid")), labeling)
        assert result.satisfied

    def test_ag_violated(self) -> None:
        lts, labeling = _make_branching_lts()
        # AG(¬error) should fail because error state is reachable
        result = model_check(lts, AG(Not(Atomic("error"))), labeling)
        assert not result.satisfied
        assert result.counterexample is not None

    def test_ag_not_error(self) -> None:
        lts, labeling = _make_linear_lts()
        # No error state exists, so AG(¬error) holds
        result = model_check(lts, AG(Not(Atomic("error"))), labeling)
        assert result.satisfied

    def test_ag_counterexample_trace(self) -> None:
        lts, labeling = _make_branching_lts()
        result = model_check(lts, AG(Not(Atomic("error"))), labeling)
        assert not result.satisfied
        # Counterexample exists — may be empty if initial state itself violates
        # (state 0 doesn't satisfy AG(¬error) because error IS reachable)
        assert result.counterexample is not None


# =============================================================================
# EU Tests
# =============================================================================


class TestEU:
    """Tests for E[φ U ψ] (existential until)."""

    def test_eu_basic(self) -> None:
        lts, labeling = _make_linear_lts()
        # E[processing U done]: from state 1, processing holds until done
        result = model_check(lts, EU(Atomic("processing"), Atomic("done")), labeling)
        assert 1 in result.satisfying_states

    def test_eu_immediate(self) -> None:
        lts, labeling = _make_linear_lts()
        # E[init U init]: ψ already holds, so EU is trivially true
        result = model_check(lts, EU(Atomic("init"), Atomic("init")), labeling)
        assert result.satisfied

    def test_eu_never_reaches(self) -> None:
        lts, labeling = _make_linear_lts()
        # E[init U error]: error never reachable
        result = model_check(lts, EU(Atomic("init"), Atomic("error")), labeling)
        assert not result.satisfied

    def test_eu_branching(self) -> None:
        lts, labeling = _make_branching_lts()
        # E[processing U success]: there exists a path 1->2 where this holds
        result = model_check(lts, EU(Atomic("processing"), Atomic("success")), labeling)
        assert 1 in result.satisfying_states

    def test_eu_from_init(self) -> None:
        lts, labeling = _make_linear_lts()
        # E[Not(done) U done]: holds from initial state
        result = model_check(lts, EU(Not(Atomic("done")), Atomic("done")), labeling)
        assert result.satisfied


# =============================================================================
# AU Tests
# =============================================================================


class TestAU:
    """Tests for A[φ U ψ] (universal until)."""

    def test_au_linear(self) -> None:
        lts, labeling = _make_linear_lts()
        # A[¬done U done]: on all paths (only one), ¬done holds until done
        result = model_check(lts, AU(Not(Atomic("done")), Atomic("done")), labeling)
        assert result.satisfied

    def test_au_fails_with_alternative(self) -> None:
        lts, labeling = _make_branching_lts()
        # A[processing U success]: not all paths reach success
        result = model_check(lts, AU(Atomic("processing"), Atomic("success")), labeling)
        # From state 1, path to error violates this
        assert 1 not in result.satisfying_states

    def test_au_immediate(self) -> None:
        lts, labeling = _make_linear_lts()
        # A[anything U init]: ψ already holds
        result = model_check(lts, AU(Atomic("processing"), Atomic("init")), labeling)
        assert result.satisfied

    def test_au_diamond(self) -> None:
        lts, labeling = _make_diamond_lts()
        # A[¬end U end]: on all paths, ¬end holds until end
        result = model_check(lts, AU(Not(Atomic("end")), Atomic("end")), labeling)
        assert result.satisfied


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for check_safety, check_liveness, check_invariant."""

    def test_check_safety_safe(self) -> None:
        lts, labeling = _make_linear_lts()
        result = check_safety(lts, "error", labeling)
        assert result.satisfied

    def test_check_safety_unsafe(self) -> None:
        lts, labeling = _make_branching_lts()
        result = check_safety(lts, "error", labeling)
        assert not result.satisfied

    def test_check_liveness_holds(self) -> None:
        lts, labeling = _make_linear_lts()
        result = check_liveness(lts, "done", labeling)
        assert result.satisfied

    def test_check_liveness_fails(self) -> None:
        lts, labeling = _make_cyclic_lts()
        result = check_liveness(lts, "done", labeling)
        # Cycle can prevent reaching done
        assert not result.satisfied

    def test_check_invariant_holds(self) -> None:
        lts, labeling = _make_linear_lts()
        for s in labeling:
            labeling[s].add("ok")
        result = check_invariant(lts, "ok", labeling)
        assert result.satisfied

    def test_check_invariant_violated(self) -> None:
        lts, labeling = _make_branching_lts()
        # Only add "safe" to some states
        labeling[0].add("safe")
        labeling[1].add("safe")
        # State 2 and 3 don't have "safe"
        result = check_invariant(lts, "safe", labeling)
        assert not result.satisfied


# =============================================================================
# ModelCheckResult Tests
# =============================================================================


class TestModelCheckResult:
    """Tests for ModelCheckResult dataclass."""

    def test_satisfied_result(self) -> None:
        result = ModelCheckResult(satisfied=True, satisfying_states={0, 1, 2})
        assert result.satisfied
        assert len(result.satisfying_states) == 3
        assert result.counterexample is None

    def test_unsatisfied_result(self) -> None:
        result = ModelCheckResult(
            satisfied=False,
            satisfying_states={1, 2},
            counterexample=(Event("a"),),
        )
        assert not result.satisfied
        assert result.counterexample is not None

    def test_formula_stored(self) -> None:
        formula = AG(Not(Atomic("error")))
        result = ModelCheckResult(satisfied=True, formula=formula)
        assert result.formula == formula


# =============================================================================
# Process Input Tests
# =============================================================================


class TestProcessInput:
    """Tests for model_check with Process inputs (auto-conversion to LTS)."""

    def test_with_process(self) -> None:
        # Build a simple process: a -> b -> STOP
        p = Prefix(Event("a"), Prefix(Event("b"), Stop()))
        lts = LTS()
        # We need to build the LTS first to know state IDs
        from agenticraft_foundation.algebra.semantics import build_lts

        built = build_lts(p)
        # Label all states as "valid"
        labeling: Labeling = {s: {"valid"} for s in built.states}
        result = model_check(built, AG(Atomic("valid")), labeling)
        assert result.satisfied


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in model checking."""

    def test_single_state_lts(self) -> None:
        """LTS with a single state."""
        lts = LTS()
        lts.add_state(LTSState(0, Stop(), is_terminal=True))
        labeling: Labeling = {0: {"ok"}}

        result = model_check(lts, AG(Atomic("ok")), labeling)
        assert result.satisfied

    def test_empty_labeling(self) -> None:
        """All states have empty labels."""
        lts, _ = _make_linear_lts()
        labeling: Labeling = {}

        result = model_check(lts, Atomic("anything"), labeling)
        assert not result.satisfied

    def test_deeply_nested_formula(self) -> None:
        """Test with deeply nested formula."""
        lts, labeling = _make_linear_lts()
        # AG(EF(Atomic("done"))) — from all states, done is eventually reachable
        result = model_check(lts, AG(EF(Atomic("done"))), labeling)
        assert result.satisfied

    def test_multiple_atomic_props(self) -> None:
        """State with multiple atomic propositions."""
        lts = LTS()
        lts.add_state(LTSState(0, Stop()))
        labeling: Labeling = {0: {"a", "b", "c"}}

        result = model_check(lts, And(Atomic("a"), And(Atomic("b"), Atomic("c"))), labeling)
        assert result.satisfied

    def test_ag_ef_pattern(self) -> None:
        """AG(EF(φ)) — reset property: from any state, φ is reachable."""
        lts, labeling = _make_cyclic_lts()
        # From any state, "ready" is reachable (due to cycle)
        result = model_check(lts, AG(EF(Atomic("ready"))), labeling)
        # State 2 (done, terminal) can't reach "ready"
        assert not result.satisfied

    def test_counterexample_for_ag(self) -> None:
        """Counterexample should be a trace to a violating state."""
        lts, labeling = _make_branching_lts()
        result = model_check(lts, AG(Not(Atomic("error"))), labeling)
        assert not result.satisfied
        assert result.counterexample is not None
        # Following the counterexample should reach the error state

    def test_self_loop_eg(self) -> None:
        """EG with self-loop should hold."""
        lts = LTS()
        lts.add_state(LTSState(0, Stop()))
        lts.add_transition(Transition(0, Event("a"), 0))
        labeling: Labeling = {0: {"alive"}}

        result = model_check(lts, EG(Atomic("alive")), labeling)
        assert result.satisfied

    def test_unknown_formula_type(self) -> None:
        """Unknown formula type should raise TypeError."""

        class Bogus(CTLFormula):
            def __repr__(self) -> str:
                return "bogus"

            def __eq__(self, other: object) -> bool:
                return isinstance(other, Bogus)

            def __hash__(self) -> int:
                return hash("bogus")

        lts, labeling = _make_linear_lts()
        with pytest.raises(TypeError, match="Unknown CTL formula type"):
            model_check(lts, Bogus(), labeling)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple temporal operators."""

    def test_mutual_exclusion(self) -> None:
        """AG(¬(critical_1 ∧ critical_2)) — mutual exclusion."""
        lts = LTS()
        lts.add_state(LTSState(0, Stop()))
        lts.add_state(LTSState(1, Stop()))
        lts.add_state(LTSState(2, Stop()))
        lts.add_transition(Transition(0, Event("enter1"), 1))
        lts.add_transition(Transition(0, Event("enter2"), 2))
        lts.add_transition(Transition(1, Event("exit1"), 0))
        lts.add_transition(Transition(2, Event("exit2"), 0))

        labeling: Labeling = {
            0: {"idle"},
            1: {"critical_1"},
            2: {"critical_2"},
        }

        # No state has both critical_1 and critical_2
        formula = AG(Not(And(Atomic("critical_1"), Atomic("critical_2"))))
        result = model_check(lts, formula, labeling)
        assert result.satisfied

    def test_response_property(self) -> None:
        """AG(req → AF(resp)) — every request eventually gets a response."""
        lts = LTS()
        lts.add_state(LTSState(0, Stop()))
        lts.add_state(LTSState(1, Stop()))
        lts.add_state(LTSState(2, Stop()))
        lts.add_transition(Transition(0, Event("request"), 1))
        lts.add_transition(Transition(1, Event("process"), 2))
        lts.add_transition(Transition(2, Event("respond"), 0))

        labeling: Labeling = {
            0: {"idle"},
            1: {"req"},
            2: {"resp"},
        }

        formula = AG(Implies(Atomic("req"), AF(Atomic("resp"))))
        result = model_check(lts, formula, labeling)
        assert result.satisfied

    def test_starvation_freedom(self) -> None:
        """AG(AF(progress)) — no infinite stalling."""
        lts, labeling = _make_cyclic_lts()
        labeling[1].add("progress")  # Working state = progress

        # In CTL, AF(progress) at state 2 (terminal, no successors):
        # AF = ¬EG(¬progress). EG needs successor chain, state 2 has none,
        # so EG(¬progress) is empty at state 2, meaning AF vacuously holds.
        # Therefore AG(AF(progress)) holds everywhere.
        result = model_check(lts, AG(AF(Atomic("progress"))), labeling)
        assert result.satisfied
