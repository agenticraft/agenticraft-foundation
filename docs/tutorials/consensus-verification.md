# Formal Consensus Verification

**Time:** 15 minutes

In this tutorial, you will model consensus states for a multi-agent system, verify classical consensus properties (agreement, validity, integrity, termination), extend to weighted consensus with quality-based agent weights, and explore MAS theory mappings for BDI and Contract Net protocols.

## Prerequisites

- Python 3.10+
- `agenticraft-foundation` installed
- Familiarity with the idea of distributed consensus (multiple agents agreeing on a value)

## What You'll Build

A consensus verification workflow for a 3-agent system. You will:

1. Define consensus states with proposals and decisions
2. Verify the four classical consensus properties
3. Model weighted consensus with agent quality weights
4. Map consensus states to MAS theory frameworks (BDI and Contract Net)

## Step 1: Define Consensus States

A consensus state captures a snapshot of a distributed agreement protocol. It records what each agent proposed, what each agent decided (if anything), and which processes are correct (not faulty).

```python
from agenticraft_foundation.specifications import (
    ConsensusState, ConsensusProperty,
    AgreementProperty, ValidityProperty,
    IntegrityProperty, TerminationProperty,
)

# Create consensus state with 3 agents
state = ConsensusState(
    proposals={"agent_0": "value_a", "agent_1": "value_a", "agent_2": "value_b"},
    decisions={"agent_0": "value_a", "agent_1": "value_a"},
    correct_processes={"agent_0", "agent_1", "agent_2"},
)
```

In this state, all three agents have made proposals. Agents 0 and 1 proposed "value_a", while agent 2 proposed "value_b". So far, agents 0 and 1 have decided on "value_a", but agent 2 has not yet decided. All three agents are marked as correct (non-faulty).

This is a realistic mid-protocol snapshot. In a running system, you would capture these states at checkpoints and verify properties to ensure the protocol is behaving correctly.

## Step 2: Verify Consensus Properties

The classical consensus properties are the foundation of distributed systems correctness. Each property captures a different aspect of what it means for agents to agree correctly.

```python
agreement = AgreementProperty()
validity = ValidityProperty()

print(f"Agreement holds: {agreement.verify(state)}")
print(f"Validity holds: {validity.verify(state)}")
```

**Agreement** requires that no two correct processes decide differently. In our state, agents 0 and 1 both decided "value_a", so agreement holds. If agent 1 had decided "value_b" instead, agreement would be violated.

**Validity** requires that any decision value was proposed by some process. Since "value_a" was proposed by agents 0 and 1, and it is the decided value, validity holds. This property prevents the protocol from deciding on a value that nobody proposed.

There are two additional properties worth understanding, even though they may not hold for every mid-protocol snapshot:

**Integrity** requires that every correct process decides at most once. This prevents flip-flopping, which would undermine the stability of the consensus.

**Termination** requires that every correct process eventually decides. In our snapshot, agent 2 has not yet decided, so termination does not hold yet -- but it should hold once the protocol completes.

## Step 3: Weighted Consensus

In many multi-agent systems, not all agents are equal. An agent backed by a more capable model, or one with a track record of higher accuracy, should have more influence on the consensus outcome. Weighted consensus formalizes this.

```python
from agenticraft_foundation.specifications import (
    WeightedConsensusState, WeightedAgreement, WeightedQuorum,
)

weighted_state = WeightedConsensusState(
    proposals={"agent_0": "value_a", "agent_1": "value_a", "agent_2": "value_b"},
    decisions={"agent_0": "value_a", "agent_1": "value_a"},
    correct_processes={"agent_0", "agent_1", "agent_2"},
    weights={"agent_0": 0.5, "agent_1": 0.3, "agent_2": 0.2},
)

weighted_agreement = WeightedAgreement()
weighted_quorum = WeightedQuorum()

print(f"Weighted agreement: {weighted_agreement.verify(weighted_state)}")
print(f"Weighted quorum: {weighted_quorum.verify(weighted_state)}")
```

The `weights` dictionary assigns a weight to each agent. Here, agent 0 has weight 0.5 (the most influential), agent 1 has 0.3, and agent 2 has 0.2. The weights should sum to 1.0.

**Weighted agreement** extends classical agreement by requiring that the decided value has sufficient weighted support. **Weighted quorum** checks whether the agents that have decided represent a sufficient fraction of the total weight. In this case, agents 0 and 1 together have weight 0.8, which is a strong quorum.

This is directly applicable to multi-agent systems where you use ensemble methods or voting among LLM agents. The weights can represent model quality scores, historical accuracy, or confidence levels.

## Step 4: MAS Theory Mappings

The consensus framework connects to broader multi-agent systems theory through mappings. These mappings translate domain-specific structures (like BDI mental states or Contract Net negotiations) into the consensus formalism, allowing you to apply the same verification properties.

```python
from agenticraft_foundation.specifications import BDIMapping, ContractNetMapping

bdi = BDIMapping()
bdi_state = bdi.map_to_mas({
    "beliefs": {"context": "current_state"},
    "desires": {"objective": "complete_task"},
    "intentions": {"assignment": "agent_0"},
})
print(f"BDI mapping: {bdi_state}")

contract_net = ContractNetMapping()
cn_state = contract_net.map_to_mas({
    "cfp": "task_description",
    "bids": {"agent_0": 0.9, "agent_1": 0.7},
    "award": "agent_0",
})
print(f"Contract Net mapping: {cn_state}")
```

The **BDI (Belief-Desire-Intention) mapping** translates the classic BDI agent architecture into MAS theory terms. Beliefs represent the agent's model of the world, desires represent its goals, and intentions represent its committed plans. The mapping produces a structured state that can be analyzed for consistency and completeness.

The **Contract Net mapping** translates a Contract Net Protocol interaction -- where a manager issues a call for proposals (CFP), agents bid, and the manager awards the task -- into a consensus-compatible structure. The bids can be mapped to proposals, and the award can be mapped to a decision, allowing you to verify properties like validity (the awarded agent actually bid) and agreement (only one agent was awarded).

These mappings are extensible. If your multi-agent system uses a different coordination pattern (auction, marketplace, negotiation), you can create custom mappings that translate your domain-specific state into the consensus formalism and reuse all the verification properties.

## Complete Script

```python
"""Consensus Verification Tutorial - Complete Script

Models consensus states, verifies properties, uses weighted consensus,
and explores MAS theory mappings.
"""
from agenticraft_foundation.specifications import (
    ConsensusState, ConsensusProperty,
    AgreementProperty, ValidityProperty,
    IntegrityProperty, TerminationProperty,
    WeightedConsensusState, WeightedAgreement, WeightedQuorum,
    BDIMapping, ContractNetMapping,
)

# Step 1: Define consensus state
state = ConsensusState(
    proposals={"agent_0": "value_a", "agent_1": "value_a", "agent_2": "value_b"},
    decisions={"agent_0": "value_a", "agent_1": "value_a"},
    correct_processes={"agent_0", "agent_1", "agent_2"},
)

# Step 2: Verify consensus properties
agreement = AgreementProperty()
validity = ValidityProperty()

print(f"Agreement holds: {agreement.verify(state)}")
print(f"Validity holds: {validity.verify(state)}")

# Step 3: Weighted consensus
weighted_state = WeightedConsensusState(
    proposals={"agent_0": "value_a", "agent_1": "value_a", "agent_2": "value_b"},
    decisions={"agent_0": "value_a", "agent_1": "value_a"},
    correct_processes={"agent_0", "agent_1", "agent_2"},
    weights={"agent_0": 0.5, "agent_1": 0.3, "agent_2": 0.2},
)

weighted_agreement = WeightedAgreement()
weighted_quorum = WeightedQuorum()

print(f"Weighted agreement: {weighted_agreement.verify(weighted_state)}")
print(f"Weighted quorum: {weighted_quorum.verify(weighted_state)}")

# Step 4: MAS theory mappings
bdi = BDIMapping()
bdi_state = bdi.map_to_mas({
    "beliefs": {"context": "current_state"},
    "desires": {"objective": "complete_task"},
    "intentions": {"assignment": "agent_0"},
})
print(f"BDI mapping: {bdi_state}")

contract_net = ContractNetMapping()
cn_state = contract_net.map_to_mas({
    "cfp": "task_description",
    "bids": {"agent_0": 0.9, "agent_1": 0.7},
    "award": "agent_0",
})
print(f"Contract Net mapping: {cn_state}")
```

## Next Steps

- Read [Consensus Concepts](../concepts/consensus.md) for the formal definitions and proofs behind these properties
- Explore the [Specifications API Reference](../api/specifications/index.md) for custom properties and verification strategies
- Revisit [Modeling Agent Coordination with CSP](csp-coordination.md) to combine process-algebraic modeling with consensus verification
