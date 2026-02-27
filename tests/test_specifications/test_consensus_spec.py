"""Tests for consensus formal specifications."""

import pytest

from agenticraft_foundation.specifications import (
    Agreement,
    ConsensusSpecification,
    ConsensusState,
    Integrity,
    PropertyStatus,
    PropertyType,
    Termination,
    Validity,
    create_byzantine_spec,
    create_crash_spec,
    hash_state,
)


class TestConsensusState:
    """Tests for ConsensusState dataclass."""

    def test_basic_state(self):
        """Test basic state creation."""
        state = ConsensusState(
            instance_id="test-1",
            participants={"a", "b", "c"},
        )

        assert state.instance_id == "test-1"
        assert len(state.participants) == 3
        assert not state.is_terminated

    def test_state_with_values(self):
        """Test state with proposed values and decisions."""
        state = ConsensusState(
            instance_id="test-2",
            participants={"a", "b", "c"},
            proposed_values={"a": 1, "b": 2, "c": 1},
            decisions={"a": 1, "b": 1},
        )

        assert state.proposed_values["a"] == 1
        assert state.decisions["a"] == 1
        assert "c" not in state.decisions


class TestAgreementProperty:
    """Tests for Agreement property."""

    def test_agreement_satisfied(self):
        """Test Agreement when all decide same value."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b", "c"},
            decisions={"a": 1, "b": 1, "c": 1},
        )

        agreement = Agreement()
        result = agreement.check(state)

        assert result.is_satisfied()
        assert result.property_type == PropertyType.SAFETY

    def test_agreement_violated(self):
        """Test Agreement when decisions differ."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b", "c"},
            decisions={"a": 1, "b": 2, "c": 1},
        )

        agreement = Agreement()
        result = agreement.check(state)

        assert not result.is_satisfied()
        assert result.status == PropertyStatus.VIOLATED
        assert result.counterexample is not None

    def test_agreement_trivial_few_decisions(self):
        """Test Agreement with < 2 decisions (trivially holds)."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b", "c"},
            decisions={"a": 1},
        )

        agreement = Agreement()
        result = agreement.check(state)

        assert result.is_satisfied()

    def test_agreement_with_correct_processes(self):
        """Test Agreement considering only correct processes."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b", "c"},
            decisions={"a": 1, "b": 2, "c": 1},  # b is faulty
        )

        # Only consider a and c as correct
        agreement = Agreement(correct_processes={"a", "c"})
        result = agreement.check(state)

        assert result.is_satisfied()


class TestValidityProperty:
    """Tests for Validity property."""

    def test_validity_satisfied(self):
        """Test Validity when decision was proposed."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b", "c"},
            proposed_values={"a": 1, "b": 2, "c": 1},
            decisions={"a": 1},
        )

        validity = Validity()
        result = validity.check(state)

        assert result.is_satisfied()

    def test_validity_violated(self):
        """Test Validity when decision wasn't proposed."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b", "c"},
            proposed_values={"a": 1, "b": 2, "c": 1},
            decisions={"a": 99},  # 99 was never proposed
        )

        validity = Validity()
        result = validity.check(state)

        assert not result.is_satisfied()
        assert result.status == PropertyStatus.VIOLATED


class TestIntegrityProperty:
    """Tests for Integrity property."""

    def test_integrity_satisfied(self):
        """Test Integrity with single decisions."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b"},
            decisions={"a": 1, "b": 2},
        )

        integrity = Integrity()
        result = integrity.check(state)

        assert result.is_satisfied()

    def test_integrity_with_history(self):
        """Test Integrity with decision history."""
        # Process 'a' decided twice with different values
        history = {"a": [1, 2]}

        state = ConsensusState(
            instance_id="test",
            participants={"a"},
            decisions={},
        )

        integrity = Integrity(decision_history=history)
        result = integrity.check(state)

        assert not result.is_satisfied()


class TestTerminationProperty:
    """Tests for Termination property."""

    def test_termination_satisfied(self):
        """Test Termination when all have decided."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b", "c"},
            decisions={"a": 1, "b": 1, "c": 1},
            is_terminated=True,
        )

        termination = Termination()
        result = termination.check(state)

        assert result.is_satisfied()

    def test_termination_in_progress(self):
        """Test Termination when still in progress."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b", "c"},
            decisions={"a": 1},  # Only a has decided
            round=5,
        )

        termination = Termination()
        result = termination.check(state)

        assert result.status == PropertyStatus.UNKNOWN

    def test_termination_timeout(self):
        """Test Termination timeout."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b", "c"},
            decisions={"a": 1},
            round=100,  # Past timeout
        )

        termination = Termination(timeout_rounds=50)
        result = termination.check(state)

        assert result.status == PropertyStatus.TIMEOUT


class TestConsensusSpecification:
    """Tests for ConsensusSpecification class."""

    def test_verify_all_properties(self):
        """Test verifying all properties."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b", "c"},
            proposed_values={"a": 1, "b": 1, "c": 1},
            decisions={"a": 1, "b": 1, "c": 1},
        )

        spec = ConsensusSpecification()
        results = spec.verify(state)

        # Should check 4 properties
        assert len(results) == 4

    def test_verify_safety_only(self):
        """Test verifying only safety properties."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b"},
            proposed_values={"a": 1, "b": 1},
            decisions={"a": 1, "b": 1},
        )

        spec = ConsensusSpecification()
        results = spec.verify_safety(state)

        # Safety properties: Agreement, Validity, Integrity
        assert all(r.property_type == PropertyType.SAFETY for r in results)

    def test_verify_liveness_only(self):
        """Test verifying only liveness properties."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b"},
            decisions={"a": 1, "b": 1},
        )

        spec = ConsensusSpecification()
        results = spec.verify_liveness(state)

        # Liveness property: Termination
        assert all(r.property_type == PropertyType.LIVENESS for r in results)

    def test_is_valid(self):
        """Test is_valid helper."""
        valid_state = ConsensusState(
            instance_id="test",
            participants={"a", "b"},
            proposed_values={"a": 1, "b": 1},
            decisions={"a": 1, "b": 1},
        )

        invalid_state = ConsensusState(
            instance_id="test",
            participants={"a", "b"},
            proposed_values={"a": 1, "b": 1},
            decisions={"a": 1, "b": 2},  # Different decisions
        )

        spec = ConsensusSpecification()

        assert spec.is_valid(valid_state)
        assert not spec.is_valid(invalid_state)

    def test_summary(self):
        """Test summary generation."""
        state = ConsensusState(
            instance_id="test-instance",
            participants={"a", "b", "c"},
            proposed_values={"a": 1, "b": 1, "c": 1},
            decisions={"a": 1, "b": 1},
            round=5,
        )

        spec = ConsensusSpecification()
        summary = spec.summary(state)

        assert "test-instance" in summary
        assert "Agreement" in summary
        assert "Validity" in summary


class TestSpecFactories:
    """Tests for specification factory functions."""

    def test_create_byzantine_spec_valid(self):
        """Test creating valid Byzantine spec."""
        spec = create_byzantine_spec(n=7, f=2)
        assert spec is not None

    def test_create_byzantine_spec_invalid(self):
        """Test creating invalid Byzantine spec raises error."""
        with pytest.raises(ValueError) as exc_info:
            create_byzantine_spec(n=5, f=2)  # Need 7 for f=2
        assert "3f+1" in str(exc_info.value)

    def test_create_crash_spec_valid(self):
        """Test creating valid crash spec."""
        spec = create_crash_spec(n=5, f=2)
        assert spec is not None

    def test_create_crash_spec_invalid(self):
        """Test creating invalid crash spec raises error."""
        with pytest.raises(ValueError) as exc_info:
            create_crash_spec(n=4, f=2)  # Need 5 for f=2
        assert "2f+1" in str(exc_info.value)


class TestHashState:
    """Tests for state hashing."""

    def test_hash_state(self):
        """Test hashing consensus state."""
        state = ConsensusState(
            instance_id="test",
            participants={"a", "b"},
            proposed_values={"a": 1},
            decisions={"a": 1},
            round=5,
        )

        hash1 = hash_state(state)
        assert len(hash1) == 16  # Truncated SHA256

    def test_hash_deterministic(self):
        """Test that same state produces same hash."""
        state1 = ConsensusState(
            instance_id="test",
            participants={"a", "b"},
            decisions={"a": 1},
        )
        state2 = ConsensusState(
            instance_id="test",
            participants={"a", "b"},
            decisions={"a": 1},
        )

        assert hash_state(state1) == hash_state(state2)

    def test_hash_different_states(self):
        """Test that different states produce different hashes."""
        state1 = ConsensusState(
            instance_id="test1",
            participants={"a", "b"},
            decisions={"a": 1},
        )
        state2 = ConsensusState(
            instance_id="test2",
            participants={"a", "b"},
            decisions={"a": 1},
        )

        assert hash_state(state1) != hash_state(state2)
