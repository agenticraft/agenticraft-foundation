"""Tests for mesh-specific complexity models.

Covers:
- LLM-specific fault models (hallucination, prompt injection, etc.)
- Mesh communication complexity bounds by topology
"""

from __future__ import annotations

from agenticraft_foundation.complexity.annotations import FaultModel
from agenticraft_foundation.complexity.bounds import MESH_COMMUNICATION_BOUNDS


class TestLLMFaultModels:
    """LLM-specific fault model extensions."""

    def test_hallucination_exists(self):
        assert FaultModel.HALLUCINATION.value == "hallucination"

    def test_prompt_injection_exists(self):
        assert FaultModel.PROMPT_INJECTION.value == "prompt_injection"

    def test_non_determinism_exists(self):
        assert FaultModel.NON_DETERMINISM.value == "non_determinism"

    def test_context_overflow_exists(self):
        assert FaultModel.CONTEXT_OVERFLOW.value == "context_overflow"

    def test_classical_models_still_present(self):
        assert FaultModel.CRASH_STOP
        assert FaultModel.CRASH_RECOVERY
        assert FaultModel.BYZANTINE
        assert FaultModel.OMISSION
        assert FaultModel.AUTHENTICATED_BYZANTINE

    def test_all_fault_models_count(self):
        # 5 classical + 4 LLM-specific = 9
        assert len(FaultModel) == 9


class TestMeshCommunicationBounds:
    """Mesh communication complexity bounds by topology type."""

    def test_bounds_dict_exists(self):
        assert isinstance(MESH_COMMUNICATION_BOUNDS, dict)
        assert len(MESH_COMMUNICATION_BOUNDS) > 0

    def test_full_mesh_bounds(self):
        key = "full_mesh_messages"
        assert key in MESH_COMMUNICATION_BOUNDS
        bound = MESH_COMMUNICATION_BOUNDS[key]
        assert "n" in bound.expression

    def test_tree_bounds(self):
        key = "tree_messages"
        assert key in MESH_COMMUNICATION_BOUNDS
        bound = MESH_COMMUNICATION_BOUNDS[key]
        assert "n" in bound.expression

    def test_ring_bounds(self):
        key = "ring_messages"
        assert key in MESH_COMMUNICATION_BOUNDS

    def test_star_bounds(self):
        key = "star_messages"
        assert key in MESH_COMMUNICATION_BOUNDS

    def test_byzantine_consensus_bound(self):
        key = "byzantine_consensus_messages"
        assert key in MESH_COMMUNICATION_BOUNDS
        bound = MESH_COMMUNICATION_BOUNDS[key]
        # Byzantine consensus is Theta(n^2)
        assert "n" in bound.expression

    def test_bounds_have_required_fields(self):
        for _key, bound in MESH_COMMUNICATION_BOUNDS.items():
            assert hasattr(bound, "expression")
            assert hasattr(bound, "problem")
