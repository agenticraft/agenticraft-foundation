"""Tests for probabilistic verification (DTMC)."""

from __future__ import annotations

import pytest

from agenticraft_foundation.verification.probabilistic import (
    DTMC,
    DTMCState,
    ExpectedStepsResult,
    ProbabilisticResult,
    ProbabilisticTransition,
    SteadyStateResult,
    build_dtmc_from_lts,
    check_reachability,
    expected_steps,
    steady_state,
)

# =============================================================================
# Helpers: Build Common DTMCs
# =============================================================================


def _make_simple_dtmc() -> DTMC:
    """Simple DTMC:
    0 --1.0--> 1 --0.9--> 2 (success, absorbing)
                1 --0.1--> 3 (error, absorbing)
    """
    dtmc = DTMC()
    dtmc.add_state(0, labels={"init"})
    dtmc.add_state(1, labels={"processing"})
    dtmc.add_state(2, labels={"success"})
    dtmc.add_state(3, labels={"error"})
    dtmc.add_transition(0, 1, probability=1.0)
    dtmc.add_transition(1, 2, probability=0.9)
    dtmc.add_transition(1, 3, probability=0.1)
    dtmc.add_transition(2, 2, probability=1.0)  # absorbing
    dtmc.add_transition(3, 3, probability=1.0)  # absorbing
    return dtmc


def _make_retry_dtmc() -> DTMC:
    """Retry DTMC: process can fail and retry.
    0 --1.0--> 1 --0.7--> 2 (success, absorbing)
                1 --0.3--> 0 (retry)
    """
    dtmc = DTMC()
    dtmc.add_state(0, labels={"init"})
    dtmc.add_state(1, labels={"attempt"})
    dtmc.add_state(2, labels={"success"})
    dtmc.add_transition(0, 1, probability=1.0)
    dtmc.add_transition(1, 2, probability=0.7)
    dtmc.add_transition(1, 0, probability=0.3)
    dtmc.add_transition(2, 2, probability=1.0)  # absorbing
    return dtmc


def _make_ergodic_dtmc() -> DTMC:
    """Ergodic (strongly connected) DTMC:
    0 --0.6--> 1
    0 --0.4--> 0
    1 --0.5--> 0
    1 --0.5--> 1
    """
    dtmc = DTMC()
    dtmc.add_state(0, labels={"a"})
    dtmc.add_state(1, labels={"b"})
    dtmc.add_transition(0, 1, probability=0.6)
    dtmc.add_transition(0, 0, probability=0.4)
    dtmc.add_transition(1, 0, probability=0.5)
    dtmc.add_transition(1, 1, probability=0.5)
    return dtmc


def _make_three_state_absorbing() -> DTMC:
    """Three-state absorbing chain:
    0 --0.5--> 1 --0.8--> 2 (absorbing)
    0 --0.5--> 0            1 --0.2--> 0
    """
    dtmc = DTMC()
    dtmc.add_state(0, labels={"start"})
    dtmc.add_state(1, labels={"middle"})
    dtmc.add_state(2, labels={"end"})
    dtmc.add_transition(0, 1, probability=0.5)
    dtmc.add_transition(0, 0, probability=0.5)
    dtmc.add_transition(1, 2, probability=0.8)
    dtmc.add_transition(1, 0, probability=0.2)
    dtmc.add_transition(2, 2, probability=1.0)  # absorbing
    return dtmc


# =============================================================================
# DTMC Construction Tests
# =============================================================================


class TestDTMC:
    """Tests for DTMC construction and validation."""

    def test_add_state(self) -> None:
        dtmc = DTMC()
        state = dtmc.add_state(0, labels={"init"})
        assert state.id == 0
        assert "init" in state.labels
        assert 0 in dtmc.states

    def test_add_transition(self) -> None:
        dtmc = DTMC()
        dtmc.add_state(0)
        dtmc.add_state(1)
        trans = dtmc.add_transition(0, 1, probability=0.5)
        assert trans.source == 0
        assert trans.target == 1
        assert trans.probability == 0.5

    def test_invalid_probability(self) -> None:
        dtmc = DTMC()
        dtmc.add_state(0)
        dtmc.add_state(1)
        with pytest.raises(ValueError, match="Probability must be in"):
            dtmc.add_transition(0, 1, probability=-0.1)
        with pytest.raises(ValueError, match="Probability must be in"):
            dtmc.add_transition(0, 1, probability=0.0)

    def test_nonexistent_state(self) -> None:
        dtmc = DTMC()
        dtmc.add_state(0)
        with pytest.raises(ValueError, match="does not exist"):
            dtmc.add_transition(0, 1, probability=0.5)

    def test_validate_valid(self) -> None:
        dtmc = _make_simple_dtmc()
        dtmc.validate()  # Should not raise

    def test_validate_missing_transitions(self) -> None:
        dtmc = DTMC()
        dtmc.add_state(0)
        with pytest.raises(ValueError, match="no outgoing transitions"):
            dtmc.validate()

    def test_validate_probability_sum(self) -> None:
        dtmc = DTMC()
        dtmc.add_state(0)
        dtmc.add_state(1)
        dtmc.add_transition(0, 1, probability=0.5)
        # Missing: should sum to 1.0
        dtmc.add_transition(1, 1, probability=1.0)
        with pytest.raises(ValueError, match="sum to"):
            dtmc.validate()

    def test_num_states(self) -> None:
        dtmc = _make_simple_dtmc()
        assert dtmc.num_states == 4

    def test_num_transitions(self) -> None:
        dtmc = _make_simple_dtmc()
        assert dtmc.num_transitions == 5

    def test_successors(self) -> None:
        dtmc = _make_simple_dtmc()
        succs = dtmc.successors(1)
        assert len(succs) == 2
        targets = {t for t, _ in succs}
        assert targets == {2, 3}

    def test_is_absorbing(self) -> None:
        dtmc = _make_simple_dtmc()
        assert dtmc.is_absorbing(2)
        assert dtmc.is_absorbing(3)
        assert not dtmc.is_absorbing(0)
        assert not dtmc.is_absorbing(1)

    def test_states_with_label(self) -> None:
        dtmc = _make_simple_dtmc()
        assert dtmc.states_with_label("success") == {2}
        assert dtmc.states_with_label("error") == {3}
        assert dtmc.states_with_label("nonexistent") == set()

    def test_states_with_labels(self) -> None:
        dtmc = _make_simple_dtmc()
        result = dtmc.states_with_labels({"success", "error"})
        assert result == {2, 3}

    def test_get_transition_probability(self) -> None:
        dtmc = _make_simple_dtmc()
        assert dtmc.get_transition_probability(1, 2) == 0.9
        assert dtmc.get_transition_probability(1, 3) == 0.1
        assert dtmc.get_transition_probability(0, 3) == 0.0


# =============================================================================
# DTMCState Tests
# =============================================================================


class TestDTMCState:
    """Tests for DTMCState."""

    def test_basic(self) -> None:
        state = DTMCState(id=0, labels={"init"})
        assert state.id == 0
        assert "init" in state.labels

    def test_equality(self) -> None:
        s1 = DTMCState(id=0)
        s2 = DTMCState(id=0, labels={"x"})
        s3 = DTMCState(id=1)
        assert s1 == s2  # Same ID
        assert s1 != s3

    def test_hash(self) -> None:
        s1 = DTMCState(id=0)
        s2 = DTMCState(id=0)
        assert hash(s1) == hash(s2)


# =============================================================================
# ProbabilisticTransition Tests
# =============================================================================


class TestProbabilisticTransition:
    """Tests for ProbabilisticTransition."""

    def test_repr(self) -> None:
        t = ProbabilisticTransition(0, 1, 0.5)
        assert "0.5000" in repr(t)

    def test_frozen(self) -> None:
        t = ProbabilisticTransition(0, 1, 0.5)
        with pytest.raises(AttributeError):
            t.probability = 0.9  # type: ignore[misc]


# =============================================================================
# Reachability Tests
# =============================================================================


class TestCheckReachability:
    """Tests for check_reachability."""

    def test_simple_reachability(self) -> None:
        dtmc = _make_simple_dtmc()
        result = check_reachability(dtmc, target_labels={"error"})
        assert abs(result.probability - 0.1) < 1e-6
        assert abs(result.per_state[0] - 0.1) < 1e-6
        assert abs(result.per_state[1] - 0.1) < 1e-6
        assert abs(result.per_state[2] - 0.0) < 1e-6
        assert abs(result.per_state[3] - 1.0) < 1e-6

    def test_success_reachability(self) -> None:
        dtmc = _make_simple_dtmc()
        result = check_reachability(dtmc, target_labels={"success"})
        assert abs(result.probability - 0.9) < 1e-6

    def test_certain_reachability(self) -> None:
        dtmc = _make_simple_dtmc()
        result = check_reachability(dtmc, target_labels={"processing"})
        # State 1 has "processing", reachable with prob 1.0 from state 0
        assert result.is_certain or abs(result.probability - 1.0) < 1e-6

    def test_impossible_reachability(self) -> None:
        dtmc = _make_simple_dtmc()
        result = check_reachability(dtmc, target_labels={"nonexistent"})
        assert result.is_impossible
        assert result.probability == 0.0

    def test_retry_reachability(self) -> None:
        dtmc = _make_retry_dtmc()
        result = check_reachability(dtmc, target_labels={"success"})
        # With retries, success is certain eventually
        assert abs(result.probability - 1.0) < 1e-6

    def test_with_target_states(self) -> None:
        dtmc = _make_simple_dtmc()
        result = check_reachability(dtmc, target_states={3})
        assert abs(result.probability - 0.1) < 1e-6

    def test_with_both_labels_and_states(self) -> None:
        dtmc = _make_simple_dtmc()
        result = check_reachability(dtmc, target_labels={"error"}, target_states={2})
        # Union: targets = {2, 3}
        assert abs(result.probability - 1.0) < 1e-6

    def test_no_target_raises(self) -> None:
        dtmc = _make_simple_dtmc()
        with pytest.raises(ValueError, match="Must specify"):
            check_reachability(dtmc)

    def test_absorbing_initial(self) -> None:
        """Initial state is already the target."""
        dtmc = DTMC()
        dtmc.add_state(0, labels={"target"})
        dtmc.add_transition(0, 0, probability=1.0)
        result = check_reachability(dtmc, target_labels={"target"})
        assert result.is_certain

    def test_three_state_chain(self) -> None:
        dtmc = _make_three_state_absorbing()
        result = check_reachability(dtmc, target_labels={"end"})
        # State 2 is absorbing with "end", reachable from all states
        assert abs(result.probability - 1.0) < 1e-6
        assert abs(result.per_state[2] - 1.0) < 1e-6


# =============================================================================
# ProbabilisticResult Tests
# =============================================================================


class TestProbabilisticResult:
    """Tests for ProbabilisticResult properties."""

    def test_is_certain(self) -> None:
        result = ProbabilisticResult(probability=1.0)
        assert result.is_certain
        assert not result.is_impossible

    def test_is_impossible(self) -> None:
        result = ProbabilisticResult(probability=0.0)
        assert result.is_impossible
        assert not result.is_certain

    def test_intermediate(self) -> None:
        result = ProbabilisticResult(probability=0.5)
        assert not result.is_certain
        assert not result.is_impossible


# =============================================================================
# Steady-State Distribution Tests
# =============================================================================


class TestSteadyState:
    """Tests for steady_state."""

    def test_absorbing_chain(self) -> None:
        dtmc = _make_simple_dtmc()
        result = steady_state(dtmc)
        # Absorbing chain: all mass ends up in absorbing states
        assert abs(result.distribution[0] - 0.0) < 1e-4
        assert abs(result.distribution[1] - 0.0) < 1e-4
        assert abs(result.distribution[2] - 0.9) < 1e-4
        assert abs(result.distribution[3] - 0.1) < 1e-4

    def test_ergodic_chain(self) -> None:
        dtmc = _make_ergodic_dtmc()
        result = steady_state(dtmc)
        assert result.converged
        # Steady-state: π(0) = 5/11 ≈ 0.4545, π(1) = 6/11 ≈ 0.5454
        # From detailed balance: π(0)*0.6 = π(1)*0.5 → π(0)/π(1) = 5/6
        assert abs(result.distribution[0] - 5 / 11) < 1e-3
        assert abs(result.distribution[1] - 6 / 11) < 1e-3

    def test_distribution_sums_to_one(self) -> None:
        dtmc = _make_ergodic_dtmc()
        result = steady_state(dtmc)
        total = sum(result.distribution.values())
        assert abs(total - 1.0) < 1e-6

    def test_absorbing_distribution_sums_to_one(self) -> None:
        dtmc = _make_simple_dtmc()
        result = steady_state(dtmc)
        total = sum(result.distribution.values())
        assert abs(total - 1.0) < 1e-6

    def test_retry_chain_converges(self) -> None:
        dtmc = _make_retry_dtmc()
        result = steady_state(dtmc)
        # All mass should eventually reach success (state 2)
        assert abs(result.distribution[2] - 1.0) < 1e-3

    def test_convergence_flag(self) -> None:
        dtmc = _make_ergodic_dtmc()
        result = steady_state(dtmc)
        assert result.converged
        assert result.iterations > 0


# =============================================================================
# SteadyStateResult Tests
# =============================================================================


class TestSteadyStateResult:
    """Tests for SteadyStateResult."""

    def test_basic(self) -> None:
        result = SteadyStateResult(
            distribution={0: 0.5, 1: 0.5},
            converged=True,
            iterations=100,
        )
        assert result.converged
        assert result.iterations == 100


# =============================================================================
# Expected Steps Tests
# =============================================================================


class TestExpectedSteps:
    """Tests for expected_steps."""

    def test_one_step(self) -> None:
        """Deterministic: 0 --1.0--> 1 (target)."""
        dtmc = DTMC()
        dtmc.add_state(0, labels={"start"})
        dtmc.add_state(1, labels={"end"})
        dtmc.add_transition(0, 1, probability=1.0)
        dtmc.add_transition(1, 1, probability=1.0)

        result = expected_steps(dtmc, target_labels={"end"})
        assert abs(result.expected - 1.0) < 1e-6
        assert abs(result.per_state[0] - 1.0) < 1e-6
        assert abs(result.per_state[1] - 0.0) < 1e-6

    def test_two_steps(self) -> None:
        """Deterministic: 0 --1.0--> 1 --1.0--> 2 (target)."""
        dtmc = DTMC()
        dtmc.add_state(0, labels={"start"})
        dtmc.add_state(1, labels={"mid"})
        dtmc.add_state(2, labels={"end"})
        dtmc.add_transition(0, 1, probability=1.0)
        dtmc.add_transition(1, 2, probability=1.0)
        dtmc.add_transition(2, 2, probability=1.0)

        result = expected_steps(dtmc, target_labels={"end"})
        assert abs(result.expected - 2.0) < 1e-6

    def test_retry_expected_steps(self) -> None:
        """With retry: E[steps] = 1/(p) from geometric distribution.

        0 --1.0--> 1 --0.7--> 2 (target)
                    1 --0.3--> 0

        E[from 0] = 1 + E[from 1]
        E[from 1] = 1 + 0.3 * E[from 0]
        Solving: E[from 1] = 1 + 0.3*(1 + E[from 1]) = 1.3 + 0.3*E[from 1]
        E[from 1] = 1.3/0.7 ≈ 1.857
        E[from 0] = 1 + 1.857 ≈ 2.857
        """
        dtmc = _make_retry_dtmc()
        result = expected_steps(dtmc, target_labels={"success"})
        assert abs(result.expected - (1 + 1.3 / 0.7)) < 1e-4
        assert not result.is_infinite

    def test_unreachable_target(self) -> None:
        """Target is unreachable — expected steps should be infinite."""
        dtmc = DTMC()
        dtmc.add_state(0, labels={"a"})
        dtmc.add_state(1, labels={"b"})
        dtmc.add_transition(0, 0, probability=1.0)
        dtmc.add_transition(1, 1, probability=1.0)

        result = expected_steps(dtmc, target_labels={"b"})
        assert result.is_infinite

    def test_already_at_target(self) -> None:
        """Initial state is the target."""
        dtmc = DTMC()
        dtmc.add_state(0, labels={"target"})
        dtmc.add_transition(0, 0, probability=1.0)

        result = expected_steps(dtmc, target_labels={"target"})
        assert abs(result.expected - 0.0) < 1e-6

    def test_no_target_raises(self) -> None:
        dtmc = _make_simple_dtmc()
        with pytest.raises(ValueError, match="Must specify"):
            expected_steps(dtmc)

    def test_with_target_states(self) -> None:
        dtmc = _make_simple_dtmc()
        result = expected_steps(dtmc, target_states={2, 3})
        # From 0: 1 step to 1, 1 step to 2 or 3 = 2 total
        assert abs(result.expected - 2.0) < 1e-6


# =============================================================================
# ExpectedStepsResult Tests
# =============================================================================


class TestExpectedStepsResult:
    """Tests for ExpectedStepsResult."""

    def test_finite(self) -> None:
        result = ExpectedStepsResult(expected=5.0)
        assert not result.is_infinite

    def test_infinite(self) -> None:
        result = ExpectedStepsResult(expected=float("inf"))
        assert result.is_infinite


# =============================================================================
# Builder Tests
# =============================================================================


class TestBuildDTMCFromLTS:
    """Tests for build_dtmc_from_lts."""

    def test_basic_build(self) -> None:
        states = {
            0: {"init"},
            1: {"done"},
        }
        transitions = {
            0: [(1, 1.0)],
            1: [(1, 1.0)],
        }
        dtmc = build_dtmc_from_lts(states, transitions)
        assert dtmc.num_states == 2
        assert dtmc.num_transitions == 2

    def test_build_validates(self) -> None:
        """Builder should validate the resulting DTMC."""
        states = {0: {"init"}, 1: {"done"}}
        transitions = {
            0: [(1, 0.5)],  # Doesn't sum to 1
            1: [(1, 1.0)],
        }
        with pytest.raises(ValueError, match="sum to"):
            build_dtmc_from_lts(states, transitions)

    def test_custom_initial(self) -> None:
        states = {0: {"a"}, 1: {"b"}}
        transitions = {0: [(0, 1.0)], 1: [(1, 1.0)]}
        dtmc = build_dtmc_from_lts(states, transitions, initial_state=1)
        assert dtmc.initial_state == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for probabilistic verification."""

    def test_agent_success_probability(self) -> None:
        """Model an agent that tries 3 times with 70% success rate.

        States: 0(try1), 1(try2), 2(try3), 3(success), 4(failure)
        """
        dtmc = DTMC()
        dtmc.add_state(0, labels={"attempt"})
        dtmc.add_state(1, labels={"attempt"})
        dtmc.add_state(2, labels={"attempt"})
        dtmc.add_state(3, labels={"success"})
        dtmc.add_state(4, labels={"failure"})

        # Try 1
        dtmc.add_transition(0, 3, probability=0.7)
        dtmc.add_transition(0, 1, probability=0.3)
        # Try 2
        dtmc.add_transition(1, 3, probability=0.7)
        dtmc.add_transition(1, 2, probability=0.3)
        # Try 3
        dtmc.add_transition(2, 3, probability=0.7)
        dtmc.add_transition(2, 4, probability=0.3)
        # Absorbing
        dtmc.add_transition(3, 3, probability=1.0)
        dtmc.add_transition(4, 4, probability=1.0)

        # P(success) = 1 - P(fail all 3) = 1 - 0.3^3 = 0.973
        result = check_reachability(dtmc, target_labels={"success"})
        assert abs(result.probability - 0.973) < 1e-6

        result_fail = check_reachability(dtmc, target_labels={"failure"})
        assert abs(result_fail.probability - 0.027) < 1e-6

        # Steady state should have P(success) = 0.973
        ss = steady_state(dtmc)
        assert abs(ss.distribution[3] - 0.973) < 1e-3
        assert abs(ss.distribution[4] - 0.027) < 1e-3

    def test_llm_fallback_chain(self) -> None:
        """Model an LLM call with fallback providers.

        Primary succeeds 95%, if fails try secondary (90%), then tertiary (85%).
        """
        dtmc = DTMC()
        dtmc.add_state(0, labels={"primary"})
        dtmc.add_state(1, labels={"secondary"})
        dtmc.add_state(2, labels={"tertiary"})
        dtmc.add_state(3, labels={"success"})
        dtmc.add_state(4, labels={"total_failure"})

        dtmc.add_transition(0, 3, probability=0.95)
        dtmc.add_transition(0, 1, probability=0.05)
        dtmc.add_transition(1, 3, probability=0.90)
        dtmc.add_transition(1, 2, probability=0.10)
        dtmc.add_transition(2, 3, probability=0.85)
        dtmc.add_transition(2, 4, probability=0.15)
        dtmc.add_transition(3, 3, probability=1.0)
        dtmc.add_transition(4, 4, probability=1.0)

        result = check_reachability(dtmc, target_labels={"success"})
        # P(success) = 1 - 0.05 * 0.10 * 0.15 = 1 - 0.00075 = 0.99925
        assert abs(result.probability - 0.99925) < 1e-6

    def test_consensus_convergence(self) -> None:
        """Model consensus: agents agree after bounded rounds.

        Simple: 0(undecided) --0.8--> 1(partial) --0.9--> 2(consensus)
                0 --0.2--> 0 (retry)
                1 --0.1--> 0 (conflict, retry)
        """
        dtmc = DTMC()
        dtmc.add_state(0, labels={"undecided"})
        dtmc.add_state(1, labels={"partial"})
        dtmc.add_state(2, labels={"consensus"})

        dtmc.add_transition(0, 1, probability=0.8)
        dtmc.add_transition(0, 0, probability=0.2)
        dtmc.add_transition(1, 2, probability=0.9)
        dtmc.add_transition(1, 0, probability=0.1)
        dtmc.add_transition(2, 2, probability=1.0)

        # Consensus is certain (all paths eventually reach it)
        result = check_reachability(dtmc, target_labels={"consensus"})
        assert abs(result.probability - 1.0) < 1e-6

        # Expected steps
        steps = expected_steps(dtmc, target_labels={"consensus"})
        assert not steps.is_infinite
        # E[from 0] = 1 + 0.2*E[from 0] + 0.8*E[from 1]
        # E[from 1] = 1 + 0.1*E[from 0]
        # E[from 0] = 1 + 0.2*E[from 0] + 0.8*(1 + 0.1*E[from 0])
        # E[from 0] = 1 + 0.2*E[from 0] + 0.8 + 0.08*E[from 0]
        # E[from 0] = 1.8 + 0.28*E[from 0]
        # 0.72*E[from 0] = 1.8 → E[from 0] = 2.5
        assert abs(steps.expected - 2.5) < 1e-4
