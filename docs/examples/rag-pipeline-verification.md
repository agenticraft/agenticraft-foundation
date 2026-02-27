# RAG Pipeline Verification

An end-to-end example that ties together **4 modules** to formally verify a multi-agent Retrieval-Augmented Generation (RAG) pipeline.

**Modules used**: `algebra` (CSP), `semantics` (LTS, deadlock), `verification` (CTL), `topology` (Laplacian)

**Run it**:

```bash
python examples/rag_pipeline_verification.py
```

---

## The Scenario

A 4-agent RAG pipeline processes user queries:

```
Router → Retriever → Processor → Responder
```

- **Router**: receives a request and forwards it
- **Retriever**: fetches relevant documents from a knowledge base
- **Processor**: transforms and ranks retrieved documents
- **Responder**: generates and delivers the final answer

The goal: prove the pipeline is deadlock-free and that every request eventually gets a response.

---

## Step 1: Model as CSP Processes

Each pipeline stage is a CSP process that performs its action and terminates:

```python
from agenticraft_foundation import Event, Prefix, Sequential, Skip, build_lts, is_deadlock_free

route = Event("route")
fetch = Event("fetch")
send_docs = Event("send_docs")
process_docs = Event("process")
respond = Event("respond")

router = Prefix(route, Skip())
retriever = Prefix(fetch, Prefix(send_docs, Skip()))
processor = Prefix(process_docs, Skip())
responder = Prefix(respond, Skip())

pipeline = Sequential(
    router,
    Sequential(retriever, Sequential(processor, responder)),
)

lts = build_lts(pipeline)
print(f"States: {lts.num_states}, Deadlock-free: {is_deadlock_free(pipeline)}")
```

The `Sequential` operator (`;` in CSP notation) chains the stages: when Router terminates (`SKIP`), Retriever starts, and so on. The healthy pipeline is deadlock-free.

---

## Step 2: Find the Bug

What if the Retriever can silently fail? In production, a vector database timeout or network partition could cause the retriever to hang indefinitely -- modeled as `STOP` (deadlock).

```python
from agenticraft_foundation import ExternalChoice, Stop, detect_deadlock

buggy_retriever = ExternalChoice(
    left=Prefix(fetch, Prefix(send_docs, Skip())),
    right=Prefix(Event("fail"), Stop()),  # Silent failure
)

buggy_pipeline = Sequential(
    router,
    Sequential(buggy_retriever, Sequential(processor, responder)),
)

buggy_lts = build_lts(buggy_pipeline)
deadlocks = detect_deadlock(buggy_lts)
print(f"Has deadlock: {deadlocks.has_deadlock}")  # True
print(f"Deadlock trace: {deadlocks.deadlock_traces[0]}")
```

`detect_deadlock` finds the exact trace leading to the deadlock state -- the path where the retriever fails and the entire pipeline gets stuck.

---

## Step 3: Fix with Timeout

The fix: wrap the Retriever with a `Timeout`. If document retrieval doesn't complete within the time bound, fall back to cached documents.

```python
from agenticraft_foundation import Timeout

fixed_retriever = Timeout(
    process=Prefix(fetch, Prefix(send_docs, Skip())),
    duration=5.0,
    fallback=Prefix(Event("cache_hit"), Skip()),
)

fixed_pipeline = Sequential(
    router,
    Sequential(fixed_retriever, Sequential(processor, responder)),
)

fixed_lts = build_lts(fixed_pipeline)
fixed_deadlocks = detect_deadlock(fixed_lts)
print(f"Deadlock-free: {not fixed_deadlocks.has_deadlock}")  # True
```

The `Timeout` operator guarantees progress: either the retriever completes normally, or the timeout fires and the fallback provides cached results. Either way, the pipeline continues.

---

## Step 4: Analyze Topology

How resilient is the agent connectivity? A linear pipeline (`Router → Retriever → Processor → Responder`) is fragile -- any single link failure disconnects the graph.

```python
from agenticraft_foundation.topology import NetworkGraph

graph = NetworkGraph()
for agent in ["router", "retriever", "processor", "responder"]:
    graph.add_node(agent)

graph.add_edge("router", "retriever")
graph.add_edge("retriever", "processor")
graph.add_edge("processor", "responder")

analysis = graph.analyze()
print(f"lambda_2: {analysis.algebraic_connectivity:.4f}")
print(f"Bottleneck edges: {analysis.bottleneck_edges}")
```

The **algebraic connectivity** ($\lambda_2$) measures how well-connected the graph is. A low $\lambda_2$ means the topology is close to being disconnected. Adding a direct `router → responder` shortcut increases $\lambda_2$ and reduces the consensus convergence bound.

---

## Step 5: CTL Model Checking

Finally, verify a temporal property: **"every request eventually gets a response"**, expressed in CTL as $\mathbf{AG}(\text{request} \Rightarrow \mathbf{AF}(\text{response}))$.

```python
from agenticraft_foundation.verification import AG, AF, Atomic, Implies, model_check

labeling = {s: set() for s in fixed_lts.states}
labeling[fixed_lts.initial_state].add("request")
for state_id in fixed_lts.states:
    if not list(fixed_lts.successors(state_id)):
        labeling[state_id].add("response")

formula = AG(Implies(Atomic("request"), AF(Atomic("response"))))
result = model_check(fixed_lts, formula, labeling)
print(f"AG(request => AF(response)): {result.satisfied}")  # True
```

The model checker exhaustively explores all reachable states and confirms: no matter which path the system takes (normal retrieval or cache fallback), a response is always eventually delivered.

---

## What This Demonstrates

| Technique | What it caught / proved |
|---|---|
| **Deadlock detection** | Silent retriever failure causes pipeline to hang |
| **Timeout operator** | Guarantees bounded progress with fallback |
| **Spectral analysis** | Linear topology is fragile; shortcuts improve resilience |
| **CTL model checking** | Every request eventually produces a response |

This is the value of formal methods for multi-agent systems: bugs that would only surface in production under specific failure conditions are caught at design time with mathematical certainty.
