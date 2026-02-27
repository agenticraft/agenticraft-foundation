"""Formal specifications for distributed protocols.

This module provides executable specifications for verifying
correctness of consensus and coordination protocols.
"""

from __future__ import annotations

from agenticraft_foundation.specifications.consensus_spec import (
    Agreement,
    ConsensusSpecification,
    ConsensusState,
    FormalProperty,
    Integrity,
    InvariantChecker,
    PropertyResult,
    PropertyStatus,
    PropertyType,
    Termination,
    Validity,
    create_byzantine_spec,
    create_crash_spec,
    hash_state,
)

# Classical MAS mappings
from agenticraft_foundation.specifications.mas_mappings import (
    BDIMapping,
    BDIState,
    ContractNetMapping,
    ContractNetPhase,
    ContractNetState,
    JointIntentionMapping,
    JointIntentionState,
    MASTheory,
    MeshState,
    SharedPlanMapping,
    SharedPlanState,
    verify_mapping_preservation,
)

# Weighted consensus
from agenticraft_foundation.specifications.weighted_consensus_spec import (
    WeightedAgreement,
    WeightedConsensusState,
    WeightedQuorum,
    WeightedValidity,
    select_weighted_leader,
)

__all__ = [
    "PropertyType",
    "PropertyStatus",
    "PropertyResult",
    "ConsensusState",
    "FormalProperty",
    "Agreement",
    "Validity",
    "Integrity",
    "Termination",
    "ConsensusSpecification",
    "InvariantChecker",
    "create_byzantine_spec",
    "create_crash_spec",
    "hash_state",
    # Weighted consensus
    "WeightedConsensusState",
    "WeightedAgreement",
    "WeightedValidity",
    "WeightedQuorum",
    "select_weighted_leader",
    # MAS mappings
    "MASTheory",
    "MeshState",
    "BDIState",
    "BDIMapping",
    "JointIntentionState",
    "JointIntentionMapping",
    "SharedPlanState",
    "SharedPlanMapping",
    "ContractNetPhase",
    "ContractNetState",
    "ContractNetMapping",
    "verify_mapping_preservation",
]
