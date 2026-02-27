"""Tests for weighted consensus specifications.

Covers:
- WeightedConsensusState
- WeightedAgreement property
- WeightedValidity property
- WeightedQuorum intersection property
- Weight-based leader selection
"""

from __future__ import annotations

import pytest

from agenticraft_foundation.specifications.consensus_spec import ConsensusState, PropertyStatus
from agenticraft_foundation.specifications.weighted_consensus_spec import (
    WeightedAgreement,
    WeightedConsensusState,
    WeightedQuorum,
    WeightedValidity,
    select_weighted_leader,
)


@pytest.fixture
def basic_weighted_state() -> WeightedConsensusState:
    """3 agents with varying weights."""
    cs = ConsensusState(
        instance_id="test",
        participants={"a1", "a2", "a3"},
        proposed_values={"a1": "x", "a2": "x", "a3": "y"},
        decisions={"a1": "x", "a2": "x"},
    )
    return WeightedConsensusState(
        consensus_state=cs,
        weights={"a1": 3.0, "a2": 2.0, "a3": 1.0},
    )


class TestWeightedConsensusState:
    def test_total_weight(self, basic_weighted_state: WeightedConsensusState):
        assert basic_weighted_state.total_weight == 6.0

    def test_quorum_threshold(self, basic_weighted_state: WeightedConsensusState):
        assert basic_weighted_state.quorum_threshold == 4.0

    def test_weight_of(self, basic_weighted_state: WeightedConsensusState):
        assert basic_weighted_state.weight_of("a1") == 3.0
        assert basic_weighted_state.weight_of("a3") == 1.0

    def test_weight_of_unknown_agent(self, basic_weighted_state: WeightedConsensusState):
        assert basic_weighted_state.weight_of("unknown") == 1.0  # default

    def test_weight_of_set(self, basic_weighted_state: WeightedConsensusState):
        assert basic_weighted_state.weight_of_set({"a1", "a2"}) == 5.0

    def test_is_quorum_sufficient(self, basic_weighted_state: WeightedConsensusState):
        assert basic_weighted_state.is_quorum({"a1", "a2"})  # 5.0 >= 4.0

    def test_is_quorum_insufficient(self, basic_weighted_state: WeightedConsensusState):
        assert not basic_weighted_state.is_quorum({"a3"})  # 1.0 < 4.0

    def test_decided_weight(self, basic_weighted_state: WeightedConsensusState):
        assert basic_weighted_state.decided_weight() == 5.0  # a1 + a2


class TestWeightedAgreement:
    def test_agreement_satisfied(self, basic_weighted_state: WeightedConsensusState):
        wa = WeightedAgreement()
        result = wa.check(
            basic_weighted_state.consensus_state,
            weighted_state=basic_weighted_state,
        )
        assert result.status == PropertyStatus.SATISFIED

    def test_agreement_violated(self):
        # Need two groups that BOTH exceed quorum threshold (2W/3).
        # 6 agents, weight 5 each → total=30, threshold=20.
        # Group {a1,a2,a3,a4} decides "x" → weight=20 ≥ 20 ✓
        # Group {a3,a4,a5,a6} decides "y" → weight=20 ≥ 20 ✓
        # But groups are keyed by decision value, so we need all-x vs all-y.
        # 6 agents, weight 4 each → total=24, threshold=16.
        # {a1,a2,a3,a4} decides "x" → weight=16 ≥ 16 ✓
        # {a5,a6,a7,a8} decides "y" → need 4 agents.
        # Simpler: use min_agreement_weight to set a low threshold.
        cs = ConsensusState(
            instance_id="test",
            participants={"a1", "a2", "a3", "a4"},
            proposed_values={"a1": "x", "a2": "y"},
            decisions={"a1": "x", "a2": "x", "a3": "y", "a4": "y"},
        )
        wcs = WeightedConsensusState(
            consensus_state=cs,
            weights={"a1": 3.0, "a2": 3.0, "a3": 3.0, "a4": 3.0},
        )
        # Total=12, default threshold=8. Each group weighs 6 < 8.
        # Use min_agreement_weight=5 so both groups (6 each) exceed it.
        wa = WeightedAgreement(min_agreement_weight=5.0)
        result = wa.check(cs, weighted_state=wcs)
        assert result.status == PropertyStatus.VIOLATED

    def test_agreement_trivial(self):
        cs = ConsensusState(
            instance_id="test",
            participants={"a1"},
            decisions={"a1": "x"},
        )
        wa = WeightedAgreement()
        result = wa.check(cs)
        assert result.status == PropertyStatus.SATISFIED


class TestWeightedValidity:
    def test_validity_satisfied(self, basic_weighted_state: WeightedConsensusState):
        wv = WeightedValidity()
        result = wv.check(
            basic_weighted_state.consensus_state,
            weighted_state=basic_weighted_state,
        )
        assert result.status == PropertyStatus.SATISFIED

    def test_validity_violated_unproposed_decision(self):
        cs = ConsensusState(
            instance_id="test",
            participants={"a1", "a2"},
            proposed_values={"a1": "x"},
            decisions={"a1": "z"},  # z was never proposed
        )
        wv = WeightedValidity()
        result = wv.check(cs)
        assert result.status == PropertyStatus.VIOLATED


class TestWeightedQuorum:
    def test_quorum_intersection_satisfied(self, basic_weighted_state: WeightedConsensusState):
        wq = WeightedQuorum(byzantine_weight=1.0)
        result = wq.check(
            basic_weighted_state.consensus_state,
            weighted_state=basic_weighted_state,
        )
        assert result.status == PropertyStatus.SATISFIED

    def test_quorum_intersection_violated(self):
        cs = ConsensusState(
            instance_id="test",
            participants={"a1", "a2"},
        )
        wcs = WeightedConsensusState(
            consensus_state=cs,
            weights={"a1": 1.0, "a2": 1.0},
        )
        # Total = 2, threshold = 4/3, intersection = 2*4/3 - 2 = 2/3 ≈ 0.67
        # Byzantine weight 1.0 > 0.67 → violated
        wq = WeightedQuorum(byzantine_weight=1.0)
        result = wq.check(cs, weighted_state=wcs)
        assert result.status == PropertyStatus.VIOLATED


class TestWeightedLeaderSelection:
    def test_leader_deterministic(self, basic_weighted_state: WeightedConsensusState):
        leader1 = select_weighted_leader(basic_weighted_state, 0)
        leader2 = select_weighted_leader(basic_weighted_state, 0)
        assert leader1 == leader2

    def test_leader_varies_by_round(self, basic_weighted_state: WeightedConsensusState):
        leaders = {select_weighted_leader(basic_weighted_state, r) for r in range(20)}
        # Should select different leaders across rounds
        assert len(leaders) >= 2

    def test_leader_from_participants(self, basic_weighted_state: WeightedConsensusState):
        for r in range(10):
            leader = select_weighted_leader(basic_weighted_state, r)
            assert leader in basic_weighted_state.consensus_state.participants

    def test_empty_participants_raises(self):
        cs = ConsensusState(instance_id="test", participants=set())
        wcs = WeightedConsensusState(consensus_state=cs)
        with pytest.raises(ValueError, match="No participants"):
            select_weighted_leader(wcs, 0)
