# Consensus & MAS Mappings

## Overview

Consensus is the fundamental problem in distributed systems: how do multiple agents agree on a single value despite failures, delays, and disagreements? AgentiCraft Foundation formalizes the four classical consensus properties, extends them with quality-weighted quorums for LLM-based agents, and provides bidirectional mappings to established multi-agent systems (MAS) theories -- BDI, Joint Intentions, SharedPlans, and Contract Net.

## Consensus Properties

The four classical consensus properties, originally formalized for distributed processes, apply directly to multi-agent coordination.

### Agreement

No two correct processes decide differently.

$$\forall \, p_i, p_j \in \text{Correct} : \text{decision}(p_i) = \text{decision}(p_j)$$

In an agent mesh, this means all non-faulty agents converge on the same answer, plan, or action.

### Validity

The decided value was proposed by some correct process.

$$\text{decision}(p_i) \in \{\text{proposal}(p_j) \mid p_j \in \text{Correct}\}$$

This prevents the consensus mechanism from inventing values -- the outcome must originate from an actual agent's proposal.

### Integrity

Every correct process decides at most once.

$$\forall \, p_i \in \text{Correct} : |\{\text{decision}(p_i)\}| \leq 1$$

Once an agent commits to a decision, it does not change its mind.

### Termination

Every correct process eventually decides.

$$\forall \, p_i \in \text{Correct} : \exists \, t : \text{decided}(p_i, t) = \text{true}$$

The protocol does not run forever -- all correct agents reach a decision in bounded time (in synchronous systems) or with probability 1 (in randomized asynchronous systems).

## Weighted Consensus

In LLM-based multi-agent systems, not all agents are equally reliable. An agent with a history of accurate responses should carry more weight than one prone to hallucination. **Weighted consensus** assigns quality weights $w_i \in [0, 1]$ to each agent $i$ based on historical reliability metrics, and requires weighted quorums rather than simple majorities.

Let $W = \sum_{i} w_i$ be the total weight of all agents.

### Weighted Agreement

No two correct weighted quorums decide differently.

$$\forall \, Q_1, Q_2 \subseteq \text{Correct} : \left(\sum_{i \in Q_1} w_i > \frac{2W}{3} \wedge \sum_{j \in Q_2} w_j > \frac{2W}{3}\right) \implies \text{decision}(Q_1) = \text{decision}(Q_2)$$

### Weighted Validity

Decided values must come from proposals with sufficient aggregate weight.

$$\text{decision} \in \left\{\text{proposal}(p_j) \;\middle|\; \sum_{p_k \in \text{supporters}(p_j)} w_k > \frac{W}{3}\right\}$$

### Weighted Quorum Intersection

Any two quorums share an honest-majority overlap, ensuring conflicting decisions are impossible.

$$\forall \, Q_1, Q_2 : \sum_{i \in Q_1 \cap Q_2 \cap \text{Correct}} w_i > \frac{W}{3}$$

This property is the weighted generalization of the classical requirement $n \geq 3f + 1$. It ensures that even if some high-weight agents are Byzantine, the overlap between any two quorums contains enough honest weight to prevent disagreement.

## MAS Theory Mappings

AgentiCraft Foundation provides bidirectional mappings between formal consensus primitives and four classical multi-agent systems theories. Each mapping preserves the formal properties of both sides -- consensus properties map to MAS invariants, and MAS constructs map to verifiable consensus configurations.

| Theory | Mapping | Formal Preservation |
|--------|---------|---------------------|
| **BDI** (Belief-Desire-Intention) | Beliefs $\leftrightarrow$ context state; Desires $\leftrightarrow$ task objectives; Intentions $\leftrightarrow$ active assignments | Intention persistence $\leftrightarrow$ consensus integrity |
| **Joint Intentions** (Cohen & Levesque) | Mutual belief $\leftrightarrow$ consensus state; Persistent goal $\leftrightarrow$ task completion condition | Joint commitment $\leftrightarrow$ agreement property |
| **SharedPlans** (Grosz & Kraus) | Recipe $\leftrightarrow$ task decomposition DAG; Subgroup plans $\leftrightarrow$ agent cluster assignments | Plan completeness $\leftrightarrow$ termination property |
| **Contract Net** (Smith 1980) | Manager broadcasts CFP $\leftrightarrow$ proposal phase; Bidders respond $\leftrightarrow$ vote phase; Manager awards $\leftrightarrow$ decision phase; Execution reports $\leftrightarrow$ commitment | Contract binding $\leftrightarrow$ validity property |

### BDI Mapping

The Belief-Desire-Intention architecture maps naturally to consensus:

- **Beliefs** are the agent's local state -- what it knows about the world and other agents. In consensus, this corresponds to the agent's view of proposed values and received messages.
- **Desires** are the agent's objectives. In a consensus context, the desire is to reach agreement on a value that satisfies the task.
- **Intentions** are the agent's committed plans. Once consensus is reached, the decided value becomes an intention that persists (integrity property).

### Joint Intentions Mapping

Cohen and Levesque's Joint Intentions theory models how agents form and maintain shared commitments:

- **Mutual belief** corresponds to the consensus state -- all agents believe the same decided value.
- **Persistent goal** corresponds to the task completion condition -- agents maintain their commitment until the goal is achieved or known to be impossible.

### SharedPlans Mapping

Grosz and Kraus's SharedPlans theory models collaborative activity through hierarchical plan structures:

- **Recipe** (the plan structure) maps to the task decomposition DAG in a workflow.
- **Subgroup plans** map to agent cluster assignments -- which subset of agents is responsible for which subtask.

### Contract Net Mapping

Smith's Contract Net Protocol maps directly to a single round of consensus:

- **Manager** broadcasts a Call For Proposals (CFP) -- analogous to the proposal phase
- **Bidders** respond with bids -- analogous to the voting phase
- **Manager** selects and awards -- analogous to the decision phase
- **Execution** and reporting -- analogous to the commitment phase

### Verification

Each mapping is verified programmatically through `verify_mapping_preservation()`, which checks that the formal properties of the source theory are maintained through the mapping and back.

## How It Maps to Code

```python
from agenticraft_foundation.specifications import (
    ConsensusSpecification, ConsensusState,
    Agreement, Validity, Integrity, Termination,
    WeightedConsensusState, WeightedAgreement, WeightedValidity,
    BDIMapping, JointIntentionMapping,
    SharedPlanMapping, ContractNetMapping,
    verify_mapping_preservation,
)

# Create a consensus specification and verify properties
spec = ConsensusSpecification()
state = ConsensusState(
    proposals={"agent-0": "plan-A", "agent-1": "plan-A", "agent-2": "plan-B"},
    decisions={"agent-0": "plan-A", "agent-1": "plan-A", "agent-2": "plan-A"},
)

# Check the four classical properties
agreement = Agreement()
validity = Validity()
assert agreement.check(state).status.name == "SATISFIED"
assert validity.check(state).status.name == "SATISFIED"

# Weighted consensus with quality-based agent weights
weighted_state = WeightedConsensusState(
    proposals={"agent-0": "plan-A", "agent-1": "plan-A", "agent-2": "plan-B"},
    decisions={"agent-0": "plan-A", "agent-1": "plan-A", "agent-2": "plan-A"},
    weights={"agent-0": 0.9, "agent-1": 0.7, "agent-2": 0.5},
)
w_agreement = WeightedAgreement()
assert w_agreement.check(weighted_state).status.name == "SATISFIED"

# Map to BDI theory and verify preservation
bdi = BDIMapping()
assert verify_mapping_preservation(bdi)
```

## Further Reading

- **API Reference**: [specifications/consensus](../api/specifications/consensus-spec.md), [specifications/mas_mappings](../api/specifications/mas-mappings.md)
- **Tutorial**: [Consensus Verification](../tutorials/consensus-verification.md)
