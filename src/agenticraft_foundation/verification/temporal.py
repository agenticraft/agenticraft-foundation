"""CTL (Computation Tree Logic) temporal model checking.

This module provides:
- CTL formula AST (Atomic, Not, And, Or, EX, EF, EG, EU, AX, AF, AG, AU)
- Bottom-up model checking via backward fixpoint computation
- Counterexample trace generation for failing properties

Theory: Clarke & Emerson (1981), Turing Award 2007.
Algorithms from Baier & Katoen, "Principles of Model Checking",
Chapters 4 (CTL) and 6 (fixpoint characterization).

CTL Semantics (over Kripke structures / LTS):
- EX φ: there Exists a neXt state satisfying φ
- AX φ: All neXt states satisfy φ
- EF φ: there Exists a path where φ holds in some Future state
- AF φ: on All paths, φ holds in some Future state
- EG φ: there Exists a path where φ holds Globally
- AG φ: on All paths, φ holds Globally
- E[φ U ψ]: there Exists a path where φ holds Until ψ
- A[φ U ψ]: on All paths, φ holds Until ψ

The model checking algorithm evaluates formulas bottom-up:
- Base case: Atomic propositions evaluated via labeling function
- Boolean: set operations (complement, intersection, union)
- EX: backward one-step (predecessors)
- EF: least fixpoint via BFS backward from sat(φ)
- EG: greatest fixpoint via iterative removal
- EU: least fixpoint combining sat(ψ) and EX
- Universal quantifiers (AX, AF, AG, AU): defined as duals of existential
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

from agenticraft_foundation.algebra.csp import Process
from agenticraft_foundation.algebra.semantics import LTS, LTSBuilder, Trace, _find_trace_to

# =============================================================================
# CTL Formula AST
# =============================================================================


class CTLFormula(ABC):
    """Abstract base class for CTL formulas."""

    @abstractmethod
    def __repr__(self) -> str: ...

    @abstractmethod
    def __eq__(self, other: object) -> bool: ...

    @abstractmethod
    def __hash__(self) -> int: ...


# --- Atomic & Boolean ---


@dataclass(frozen=True)
class Atomic(CTLFormula):
    """Atomic proposition: holds in states labeled with the given proposition.

    Attributes:
        prop: The name of the atomic proposition.
    """

    prop: str

    def __repr__(self) -> str:
        return self.prop

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Atomic) and self.prop == other.prop

    def __hash__(self) -> int:
        return hash(("Atomic", self.prop))


@dataclass(frozen=True)
class Not(CTLFormula):
    """Negation: ¬φ.

    Attributes:
        formula: The formula to negate.
    """

    formula: CTLFormula

    def __repr__(self) -> str:
        return f"¬({self.formula})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Not) and self.formula == other.formula

    def __hash__(self) -> int:
        return hash(("Not", self.formula))


@dataclass(frozen=True)
class And(CTLFormula):
    """Conjunction: φ ∧ ψ.

    Attributes:
        left: Left conjunct.
        right: Right conjunct.
    """

    left: CTLFormula
    right: CTLFormula

    def __repr__(self) -> str:
        return f"({self.left} ∧ {self.right})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, And) and self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash(("And", self.left, self.right))


@dataclass(frozen=True)
class Or(CTLFormula):
    """Disjunction: φ ∨ ψ.

    Attributes:
        left: Left disjunct.
        right: Right disjunct.
    """

    left: CTLFormula
    right: CTLFormula

    def __repr__(self) -> str:
        return f"({self.left} ∨ {self.right})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Or) and self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash(("Or", self.left, self.right))


@dataclass(frozen=True)
class Implies(CTLFormula):
    """Implication: φ → ψ (syntactic sugar for ¬φ ∨ ψ).

    Attributes:
        left: Antecedent.
        right: Consequent.
    """

    left: CTLFormula
    right: CTLFormula

    def __repr__(self) -> str:
        return f"({self.left} → {self.right})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Implies) and self.left == other.left and self.right == other.right
        )

    def __hash__(self) -> int:
        return hash(("Implies", self.left, self.right))


# --- Existential Path Quantifiers ---


@dataclass(frozen=True)
class EX(CTLFormula):
    """EX φ: there exists a next state satisfying φ.

    Attributes:
        formula: The formula that must hold in some successor.
    """

    formula: CTLFormula

    def __repr__(self) -> str:
        return f"EX({self.formula})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EX) and self.formula == other.formula

    def __hash__(self) -> int:
        return hash(("EX", self.formula))


@dataclass(frozen=True)
class EF(CTLFormula):
    """EF φ: there exists a path where φ eventually holds.

    Equivalent to E[True U φ].

    Attributes:
        formula: The formula that must eventually hold.
    """

    formula: CTLFormula

    def __repr__(self) -> str:
        return f"EF({self.formula})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EF) and self.formula == other.formula

    def __hash__(self) -> int:
        return hash(("EF", self.formula))


@dataclass(frozen=True)
class EG(CTLFormula):
    """EG φ: there exists a path where φ holds globally.

    Attributes:
        formula: The formula that must hold on all states of some path.
    """

    formula: CTLFormula

    def __repr__(self) -> str:
        return f"EG({self.formula})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EG) and self.formula == other.formula

    def __hash__(self) -> int:
        return hash(("EG", self.formula))


@dataclass(frozen=True)
class EU(CTLFormula):
    """E[φ U ψ]: there exists a path where φ holds until ψ holds.

    Attributes:
        left: The formula that must hold until right holds.
        right: The formula that must eventually hold.
    """

    left: CTLFormula
    right: CTLFormula

    def __repr__(self) -> str:
        return f"E[{self.left} U {self.right}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EU) and self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash(("EU", self.left, self.right))


# --- Universal Path Quantifiers ---


@dataclass(frozen=True)
class AX(CTLFormula):
    """AX φ: all next states satisfy φ.

    Equivalent to ¬EX(¬φ).

    Attributes:
        formula: The formula that must hold in all successors.
    """

    formula: CTLFormula

    def __repr__(self) -> str:
        return f"AX({self.formula})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AX) and self.formula == other.formula

    def __hash__(self) -> int:
        return hash(("AX", self.formula))


@dataclass(frozen=True)
class AF(CTLFormula):
    """AF φ: on all paths, φ eventually holds.

    Equivalent to ¬EG(¬φ).

    Attributes:
        formula: The formula that must eventually hold on all paths.
    """

    formula: CTLFormula

    def __repr__(self) -> str:
        return f"AF({self.formula})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AF) and self.formula == other.formula

    def __hash__(self) -> int:
        return hash(("AF", self.formula))


@dataclass(frozen=True)
class AG(CTLFormula):
    """AG φ: on all paths, φ holds globally.

    Equivalent to ¬EF(¬φ).

    Attributes:
        formula: The formula that must hold on all paths globally.
    """

    formula: CTLFormula

    def __repr__(self) -> str:
        return f"AG({self.formula})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AG) and self.formula == other.formula

    def __hash__(self) -> int:
        return hash(("AG", self.formula))


@dataclass(frozen=True)
class AU(CTLFormula):
    """A[φ U ψ]: on all paths, φ holds until ψ holds.

    Attributes:
        left: The formula that must hold until right holds.
        right: The formula that must eventually hold.
    """

    left: CTLFormula
    right: CTLFormula

    def __repr__(self) -> str:
        return f"A[{self.left} U {self.right}]"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AU) and self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash(("AU", self.left, self.right))


# =============================================================================
# Labeling
# =============================================================================

Labeling = dict[int, set[str]]
"""Mapping from state IDs to sets of atomic proposition names."""


# =============================================================================
# Model Check Result
# =============================================================================


@dataclass
class ModelCheckResult:
    """Result of CTL model checking.

    Attributes:
        satisfied: Whether the formula holds in the initial state.
        satisfying_states: Set of states where the formula holds.
        counterexample: A trace to a violating state (if not satisfied),
            found via BFS from the initial state.
        formula: The formula that was checked.
    """

    satisfied: bool
    satisfying_states: set[int] = field(default_factory=set)
    counterexample: Trace | None = None
    formula: CTLFormula | None = None


# =============================================================================
# Model Checking Algorithm
# =============================================================================


def _to_lts(p: Process | LTS) -> LTS:
    """Convert process to LTS if needed."""
    if isinstance(p, LTS):
        return p
    return LTSBuilder().build(p)


def _all_states(lts: LTS) -> set[int]:
    """Return set of all state IDs in the LTS."""
    return set(lts.states.keys())


def _pre_exists(lts: LTS, target_states: set[int]) -> set[int]:
    """Compute EX-predecessor: states with at least one successor in target_states.

    This considers all transitions (including tau) for CTL semantics,
    since the LTS models the full transition system.

    Args:
        lts: The LTS.
        target_states: States that must be reachable in one step.

    Returns:
        Set of states that have at least one successor in target_states.
    """
    result: set[int] = set()
    for state_id in lts.states:
        for _, target in lts.successors(state_id):
            if target in target_states:
                result.add(state_id)
                break
    return result


def _pre_forall(lts: LTS, target_states: set[int]) -> set[int]:
    """Compute AX-predecessor: states where ALL successors are in target_states.

    States with no successors (deadlocks) vacuously satisfy AX.

    Args:
        lts: The LTS.
        target_states: States that all successors must be in.

    Returns:
        Set of states where every successor is in target_states.
    """
    result: set[int] = set()
    for state_id in lts.states:
        successors = list(lts.successors(state_id))
        if not successors:
            # No successors — vacuously true
            result.add(state_id)
        elif all(target in target_states for _, target in successors):
            result.add(state_id)
    return result


def _sat(lts: LTS, formula: CTLFormula, labeling: Labeling) -> set[int]:
    """Compute the satisfaction set for a CTL formula.

    Bottom-up evaluation over the formula AST using fixpoint algorithms.

    Args:
        lts: The LTS (Kripke structure).
        formula: The CTL formula to evaluate.
        labeling: Mapping from state IDs to atomic proposition sets.

    Returns:
        Set of state IDs where the formula holds.
    """
    all_s = _all_states(lts)

    # --- Atomic ---
    if isinstance(formula, Atomic):
        return {s for s in all_s if formula.prop in labeling.get(s, set())}

    # --- Boolean ---
    if isinstance(formula, Not):
        return all_s - _sat(lts, formula.formula, labeling)

    if isinstance(formula, And):
        return _sat(lts, formula.left, labeling) & _sat(lts, formula.right, labeling)

    if isinstance(formula, Or):
        return _sat(lts, formula.left, labeling) | _sat(lts, formula.right, labeling)

    if isinstance(formula, Implies):
        # φ → ψ ≡ ¬φ ∨ ψ
        return (all_s - _sat(lts, formula.left, labeling)) | _sat(lts, formula.right, labeling)

    # --- EX: states with at least one successor in sat(φ) ---
    if isinstance(formula, EX):
        sat_inner = _sat(lts, formula.formula, labeling)
        return _pre_exists(lts, sat_inner)

    # --- AX: states where ALL successors satisfy φ ---
    # AX φ ≡ ¬EX(¬φ)
    if isinstance(formula, AX):
        sat_inner = _sat(lts, formula.formula, labeling)
        return _pre_forall(lts, sat_inner)

    # --- EF: least fixpoint μZ. sat(φ) ∪ EX(Z) ---
    # BFS backward from sat(φ)
    if isinstance(formula, EF):
        sat_phi = _sat(lts, formula.formula, labeling)
        return _lfp_ef(lts, sat_phi)

    # --- AF: least fixpoint μZ. sat(φ) ∪ AX(Z) ---
    # AF φ ≡ ¬EG(¬φ)
    if isinstance(formula, AF):
        sat_not_phi = all_s - _sat(lts, formula.formula, labeling)
        sat_eg_not_phi = _gfp_eg(lts, sat_not_phi)
        return all_s - sat_eg_not_phi

    # --- EG: greatest fixpoint νZ. sat(φ) ∩ EX(Z) ---
    if isinstance(formula, EG):
        sat_phi = _sat(lts, formula.formula, labeling)
        return _gfp_eg(lts, sat_phi)

    # --- AG: AG φ ≡ ¬EF(¬φ) ---
    if isinstance(formula, AG):
        sat_not_phi = all_s - _sat(lts, formula.formula, labeling)
        sat_ef_not_phi = _lfp_ef(lts, sat_not_phi)
        return all_s - sat_ef_not_phi

    # --- EU: least fixpoint μZ. sat(ψ) ∪ (sat(φ) ∩ EX(Z)) ---
    if isinstance(formula, EU):
        sat_phi = _sat(lts, formula.left, labeling)
        sat_psi = _sat(lts, formula.right, labeling)
        return _lfp_eu(lts, sat_phi, sat_psi)

    # --- AU: A[φ U ψ] ≡ ¬E[¬ψ U (¬φ ∧ ¬ψ)] ∧ ¬EG(¬ψ) ---
    if isinstance(formula, AU):
        sat_phi = _sat(lts, formula.left, labeling)
        sat_psi = _sat(lts, formula.right, labeling)
        return _sat_au(lts, sat_phi, sat_psi, all_s)

    raise TypeError(f"Unknown CTL formula type: {type(formula).__name__}")


def _lfp_ef(lts: LTS, target: set[int]) -> set[int]:
    """Least fixpoint for EF: backward BFS from target states.

    EF φ = μZ. sat(φ) ∪ EX(Z)

    Args:
        lts: The LTS.
        target: States where φ holds (the initial seed).

    Returns:
        States from which target is reachable.
    """
    result = set(target)
    queue = deque(target)

    while queue:
        state_id = queue.popleft()
        # Add all predecessors
        for _, pred in lts.predecessors(state_id):
            if pred not in result:
                result.add(pred)
                queue.append(pred)

    return result


def _gfp_eg(lts: LTS, sat_phi: set[int]) -> set[int]:
    """Greatest fixpoint for EG: iterative removal.

    EG φ = νZ. sat(φ) ∩ EX(Z)

    Start with sat(φ), iteratively remove states that have no successor in
    the current set.

    Args:
        lts: The LTS.
        sat_phi: States where φ holds.

    Returns:
        States where EG φ holds.
    """
    current = set(sat_phi)

    changed = True
    while changed:
        changed = False
        to_remove: set[int] = set()
        for state_id in current:
            # Check if there's at least one successor still in current
            has_successor_in_set = any(
                target in current for _, target in lts.successors(state_id)
            )
            if not has_successor_in_set:
                to_remove.add(state_id)

        if to_remove:
            current -= to_remove
            changed = True

    return current


def _lfp_eu(lts: LTS, sat_phi: set[int], sat_psi: set[int]) -> set[int]:
    """Least fixpoint for EU: backward BFS.

    E[φ U ψ] = μZ. sat(ψ) ∪ (sat(φ) ∩ EX(Z))

    Start from sat(ψ), add predecessors that satisfy φ.

    Args:
        lts: The LTS.
        sat_phi: States where φ holds.
        sat_psi: States where ψ holds.

    Returns:
        States where E[φ U ψ] holds.
    """
    result = set(sat_psi)
    queue = deque(sat_psi)

    while queue:
        state_id = queue.popleft()
        for _, pred in lts.predecessors(state_id):
            if pred not in result and pred in sat_phi:
                result.add(pred)
                queue.append(pred)

    return result


def _sat_au(
    lts: LTS,
    sat_phi: set[int],
    sat_psi: set[int],
    all_states: set[int],
) -> set[int]:
    """Compute A[φ U ψ] satisfaction set.

    A[φ U ψ] = μZ. sat(ψ) ∪ (sat(φ) ∩ AX(Z))

    Iterative least fixpoint: start with sat(ψ), then add states where
    φ holds and ALL successors are already in the set.

    Args:
        lts: The LTS.
        sat_phi: States where φ holds.
        sat_psi: States where ψ holds.
        all_states: All states in the LTS.

    Returns:
        States where A[φ U ψ] holds.
    """
    result = set(sat_psi)

    changed = True
    while changed:
        changed = False
        for state_id in all_states:
            if state_id in result:
                continue
            if state_id not in sat_phi:
                continue
            # Check AX(result): all successors in result
            successors = list(lts.successors(state_id))
            if not successors:
                # No successors — φ holds but ψ doesn't, and no path continues
                # A[φ U ψ] requires ψ to eventually hold, so this state doesn't satisfy
                continue
            if all(target in result for _, target in successors):
                result.add(state_id)
                changed = True

    return result


def _find_counterexample_trace(
    lts: LTS,
    satisfying: set[int],
    max_length: int = 100,
) -> Trace | None:
    """Find a trace from the initial state to a non-satisfying state.

    Used to generate counterexamples for failing CTL properties.

    Args:
        lts: The LTS.
        satisfying: States that satisfy the property.
        max_length: Maximum trace length.

    Returns:
        A trace leading to a non-satisfying state, or None if all
        reachable states satisfy the property.
    """
    violating = set(lts.states.keys()) - satisfying
    if not violating:
        return None

    # Find shortest trace to any violating state
    for target in violating:
        trace = _find_trace_to(lts, target, max_length)
        if trace is not None:
            return trace

    return None


# =============================================================================
# Public API
# =============================================================================


def model_check(
    system: Process | LTS,
    formula: CTLFormula,
    labeling: Labeling,
    max_counterexample_length: int = 100,
) -> ModelCheckResult:
    """Check whether a CTL formula holds in the initial state of a system.

    Performs bottom-up evaluation of the formula over the LTS state space.

    Args:
        system: The system to check (CSP process or LTS).
        formula: The CTL formula to verify.
        labeling: Mapping from state IDs to sets of atomic propositions.
        max_counterexample_length: Maximum length for counterexample traces.

    Returns:
        ModelCheckResult with satisfaction status, satisfying states,
        and counterexample trace if the property fails.

    Example:
        >>> labeling = {0: {"init"}, 1: {"processing"}, 2: {"done"}}
        >>> result = model_check(lts, AG(Not(Atomic("error"))), labeling)
        >>> result.satisfied
        True
    """
    lts = _to_lts(system)
    sat_set = _sat(lts, formula, labeling)

    satisfied = lts.initial_state in sat_set

    counterexample: Trace | None = None
    if not satisfied:
        counterexample = _find_counterexample_trace(
            lts, sat_set, max_counterexample_length
        )

    return ModelCheckResult(
        satisfied=satisfied,
        satisfying_states=sat_set,
        counterexample=counterexample,
        formula=formula,
    )


def check_safety(
    system: Process | LTS,
    bad_prop: str,
    labeling: Labeling,
) -> ModelCheckResult:
    """Check a safety property: AG(¬bad_prop).

    Convenience function for the common pattern of verifying that a
    "bad" state is never reached.

    Args:
        system: The system to check.
        bad_prop: Atomic proposition labeling "bad" states.
        labeling: State labeling function.

    Returns:
        ModelCheckResult — satisfied means the bad state is unreachable.
    """
    return model_check(system, AG(Not(Atomic(bad_prop))), labeling)


def check_liveness(
    system: Process | LTS,
    good_prop: str,
    labeling: Labeling,
) -> ModelCheckResult:
    """Check a liveness property: AF(good_prop).

    Convenience function for verifying that a "good" state is eventually
    reached on all paths.

    Args:
        system: The system to check.
        good_prop: Atomic proposition labeling "good" states.
        labeling: State labeling function.

    Returns:
        ModelCheckResult — satisfied means the good state is always eventually reached.
    """
    return model_check(system, AF(Atomic(good_prop)), labeling)


def check_invariant(
    system: Process | LTS,
    invariant_prop: str,
    labeling: Labeling,
) -> ModelCheckResult:
    """Check an invariant property: AG(invariant_prop).

    Convenience function for verifying that a property holds in all
    reachable states.

    Args:
        system: The system to check.
        invariant_prop: Atomic proposition that should hold everywhere.
        labeling: State labeling function.

    Returns:
        ModelCheckResult — satisfied means the invariant holds globally.
    """
    return model_check(system, AG(Atomic(invariant_prop)), labeling)


__all__ = [
    # Formula AST
    "CTLFormula",
    "Atomic",
    "Not",
    "And",
    "Or",
    "Implies",
    "EX",
    "EF",
    "EG",
    "EU",
    "AX",
    "AF",
    "AG",
    "AU",
    # Types
    "Labeling",
    # Result
    "ModelCheckResult",
    # Model checking
    "model_check",
    "check_safety",
    "check_liveness",
    "check_invariant",
]
