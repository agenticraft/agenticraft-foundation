"""RAG Pipeline Verification -- End-to-end formal verification of a multi-agent system.

A 4-agent Retrieval-Augmented Generation pipeline:
  Router → Retriever → Processor → Responder

This example demonstrates:
  1. Modeling the pipeline as CSP processes
  2. Finding a deadlock caused by a silently-failing Retriever
  3. Fixing it with a Timeout fallback and re-verifying
  4. Analyzing the agent topology with spectral graph theory
  5. Checking a temporal property with CTL model checking

Ties together: algebra (CSP), semantics (LTS), verification (CTL), topology (Laplacian).
"""

from agenticraft_foundation import (
    Event,
    ExternalChoice,
    Prefix,
    Sequential,
    Skip,
    Stop,
    Timeout,
    build_lts,
    detect_deadlock,
    is_deadlock_free,
    traces,
)
from agenticraft_foundation.topology import NetworkGraph
from agenticraft_foundation.verification import AF, AG, Atomic, Implies, model_check

# ---------------------------------------------------------------------------
# Step 1: Define the RAG pipeline as CSP processes
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Model the RAG pipeline as CSP processes")
print("=" * 60)

# Events for each pipeline stage
route = Event("route")
fetch = Event("fetch")
send_docs = Event("send_docs")
process_docs = Event("process")
respond = Event("respond")

# Each stage: perform its action, then terminate successfully
router = Prefix(route, Skip())
retriever = Prefix(fetch, Prefix(send_docs, Skip()))
processor = Prefix(process_docs, Skip())
responder = Prefix(respond, Skip())

# Sequential pipeline: Router ; Retriever ; Processor ; Responder
pipeline = Sequential(
    router,
    Sequential(retriever, Sequential(processor, responder)),
)

lts = build_lts(pipeline)
print(f"  Pipeline states: {lts.num_states}")
print(f"  Pipeline transitions: {lts.num_transitions}")
print(f"  Deadlock-free: {is_deadlock_free(pipeline)}")

all_traces = list(traces(lts, max_length=10))
print(f"  Traces: {len(all_traces)}")
for t in all_traces[:3]:
    print(f"    {' → '.join(str(e) for e in t)}")

# ---------------------------------------------------------------------------
# Step 2: Introduce a bug -- Retriever can silently fail
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 2: Buggy pipeline -- Retriever can silently fail")
print("=" * 60)

# The buggy retriever nondeterministically either works or deadlocks.
# ExternalChoice models: the environment might send a request that
# triggers the failure path (STOP = deadlock, no progress possible).
buggy_retriever = ExternalChoice(
    left=Prefix(fetch, Prefix(send_docs, Skip())),
    right=Prefix(Event("fail"), Stop()),  # Silent failure → deadlock
)

buggy_pipeline = Sequential(
    router,
    Sequential(buggy_retriever, Sequential(processor, responder)),
)

buggy_lts = build_lts(buggy_pipeline)
deadlocks = detect_deadlock(buggy_lts)

print(f"  States: {buggy_lts.num_states}")
print(f"  Has deadlock: {deadlocks.has_deadlock}")
print(f"  Deadlock states: {len(deadlocks.deadlock_states)}")
if deadlocks.deadlock_traces:
    print(f"  Deadlock trace: {' → '.join(str(e) for e in deadlocks.deadlock_traces[0])}")

# ---------------------------------------------------------------------------
# Step 3: Fix with Timeout fallback and re-verify
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 3: Fixed pipeline -- Timeout with cache fallback")
print("=" * 60)

# Wrap the retriever with a timeout: if fetch doesn't complete,
# fall back to cached documents.
cache_hit = Event("cache_hit")
fixed_retriever = Timeout(
    process=Prefix(fetch, Prefix(send_docs, Skip())),
    duration=5.0,
    fallback=Prefix(cache_hit, Skip()),
)

fixed_pipeline = Sequential(
    router,
    Sequential(fixed_retriever, Sequential(processor, responder)),
)

fixed_lts = build_lts(fixed_pipeline)
fixed_deadlocks = detect_deadlock(fixed_lts)

print(f"  States: {fixed_lts.num_states}")
print(f"  Deadlock-free: {not fixed_deadlocks.has_deadlock}")
print(f"  Deadlock states: {len(fixed_deadlocks.deadlock_states)}")

fixed_traces = list(traces(fixed_lts, max_length=10))
print(f"  Possible traces: {len(fixed_traces)}")
for t in fixed_traces:
    print(f"    {' → '.join(str(e) for e in t)}")

# ---------------------------------------------------------------------------
# Step 4: Analyze the agent topology
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 4: Topology analysis -- spectral resilience")
print("=" * 60)

# Build a NetworkGraph representing agent connectivity
graph = NetworkGraph()
graph.add_node("router", weight=1.0)
graph.add_node("retriever", weight=1.0)
graph.add_node("processor", weight=1.0)
graph.add_node("responder", weight=1.0)

# Linear pipeline topology
graph.add_edge("router", "retriever")
graph.add_edge("retriever", "processor")
graph.add_edge("processor", "responder")

analysis = graph.analyze()
print(f"  Nodes: {analysis.num_nodes}")
print(f"  Edges: {analysis.num_edges}")
print(f"  Algebraic connectivity (lambda_2): {analysis.algebraic_connectivity:.4f}")
print(f"  Consensus bound: {analysis.consensus_bound:.2f}")
print(f"  Connected: {analysis.is_connected}")
print(f"  Bottleneck edges: {analysis.bottleneck_edges}")

# Compare: what if we add a direct router→responder link for resilience?
graph.add_edge("router", "responder")
resilient_analysis = graph.analyze()
print()
print("  After adding router→responder shortcut:")
print(f"    lambda_2: {resilient_analysis.algebraic_connectivity:.4f} "
      f"(was {analysis.algebraic_connectivity:.4f})")
print(f"    Consensus bound: {resilient_analysis.consensus_bound:.2f} "
      f"(was {analysis.consensus_bound:.2f})")

# ---------------------------------------------------------------------------
# Step 5: CTL temporal property -- "every request eventually gets a response"
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Step 5: CTL model checking -- AG(request => AF(response))")
print("=" * 60)

# Label states with atomic propositions
labeling: dict[int, set[str]] = {s: set() for s in fixed_lts.states}

# Label the initial state as "request" (a request has arrived)
labeling[fixed_lts.initial_state].add("request")

# Label terminal states as "response" (response has been delivered)
for state_id in fixed_lts.states:
    if not list(fixed_lts.successors(state_id)):
        labeling[state_id].add("response")

# Check: AG(request => AF(response))
# "In all reachable states, if a request is present, a response is eventually delivered."
formula = AG(Implies(Atomic("request"), AF(Atomic("response"))))
result = model_check(fixed_lts, formula, labeling)

print("  Formula: AG(request => AF(response))")
print(f"  Satisfied: {result.satisfied}")
print(f"  States satisfying formula: {len(result.satisfying_states)}/{fixed_lts.num_states}")

# Also check simple liveness: "response is always eventually reached"
liveness = model_check(fixed_lts, AF(Atomic("response")), labeling)
print(f"  AF(response) satisfied: {liveness.satisfied}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Summary")
print("=" * 60)
print(f"  Buggy pipeline:  {buggy_lts.num_states} states, "
      f"deadlock={deadlocks.has_deadlock}")
print(f"  Fixed pipeline:  {fixed_lts.num_states} states, "
      f"deadlock={fixed_deadlocks.has_deadlock}")
print(f"  Topology lambda_2: {resilient_analysis.algebraic_connectivity:.4f} "
      f"(resilient)")
print(f"  Temporal AG(req=>AF(resp)): {result.satisfied}")
