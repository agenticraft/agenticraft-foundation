# Verifying Protocols with Session Types

**Time:** 15 minutes

In this tutorial, you will define a multi-party session type for a RAG (Retrieval-Augmented Generation) pipeline, project it to local types for each role, verify well-formedness, and set up runtime session monitoring.

## Prerequisites

- Python 3.10+
- `agenticraft-foundation` installed
- Familiarity with the concept of communication protocols (who sends what to whom)

## What You'll Build

A RAG pipeline protocol with 3 roles -- client, retriever, and generator -- modeled as a multiparty session type. You will:

1. Define the global protocol as a sequence of interactions
2. Project the global type to each participant's local view
3. Verify that the protocol is well-formed (no stuck participants, no orphan messages)
4. Set up runtime monitoring to detect protocol violations
5. Use built-in patterns for common protocol shapes

## Step 1: Define the Global Type

A global type describes a protocol from a bird's-eye view: who sends what to whom, and in what order. The RAG pipeline has three interactions:

1. The client sends a query to the retriever
2. The retriever sends context to the generator
3. The generator sends a response back to the client

```python
from agenticraft_foundation.mpst import GlobalInteraction, GlobalType

rag_protocol = GlobalType(interactions=[
    GlobalInteraction(sender="client", receiver="retriever", message_type="Query"),
    GlobalInteraction(sender="retriever", receiver="generator", message_type="Context"),
    GlobalInteraction(sender="generator", receiver="client", message_type="Response"),
])
```

Each `GlobalInteraction` specifies a sender, a receiver, and a message type. The interactions are ordered -- the query must happen before the context retrieval, which must happen before the response generation. This ordering is what makes the protocol a pipeline rather than a free-for-all.

## Step 2: Project to Local Types

Each participant in the protocol only sees their own sends and receives. Projection takes the global type and produces a local type for each role. This is a fundamental operation in multiparty session type theory -- it ensures that each participant has a consistent view of the protocol.

```python
from agenticraft_foundation.mpst import Projector

projector = Projector()
local_client = projector.project(rag_protocol, "client")
local_retriever = projector.project(rag_protocol, "retriever")
local_generator = projector.project(rag_protocol, "generator")

print(f"Client view: {local_client}")
print(f"Retriever view: {local_retriever}")
print(f"Generator view: {local_generator}")
```

The client's local type will show: send `Query` to retriever, then receive `Response` from generator. The retriever sees: receive `Query` from client, then send `Context` to generator. The generator sees: receive `Context` from retriever, then send `Response` to client.

Notice that projection preserves the causal ordering. The client knows it must send before it receives. The generator knows it must receive before it sends. This is exactly the information each participant needs to implement the protocol correctly.

## Step 3: Verify Well-Formedness

A well-formed protocol guarantees several properties: no participant gets stuck waiting for a message that never arrives, no messages are sent to participants who are not expecting them, and every branch of a choice is projectable to all roles.

```python
from agenticraft_foundation.mpst import SessionTypeChecker

checker = SessionTypeChecker()
result = checker.check_well_formedness(rag_protocol)
print(f"Well-formed: {result.is_valid}")
if not result.is_valid:
    for error in result.errors:
        print(f"  Error: {error}")
```

For our RAG protocol, well-formedness should pass. But consider what happens if you add an interaction where the generator sends directly to the retriever without the client's knowledge -- the checker would flag this as a potential issue depending on the protocol structure.

Well-formedness checking catches design errors before any code is written. In a multi-agent system with dozens of roles and message types, this verification is essential.

## Step 4: Set Up Session Monitoring

Once you have verified the protocol at design time, you can monitor it at runtime. A session monitor tracks the actual messages exchanged and checks them against the local type.

```python
from agenticraft_foundation.mpst import SessionMonitor

monitor = SessionMonitor(local_type=local_client)
# Simulate message exchange
monitor.send("Query", to="retriever")
monitor.receive("Response", from_role="generator")
print(f"Session complete: {monitor.is_complete()}")
```

The monitor tracks the current state of the session. After the client sends a `Query` and receives a `Response`, the session is complete according to the local type. If the client tried to send a second `Query` without the protocol allowing it, the monitor would raise a violation.

In a production system, you would wrap your actual message-passing code with monitor calls. This gives you runtime guarantees that agents are following the protocol, without modifying the agents themselves.

## Step 5: Use Built-In Patterns

Many protocols follow common shapes. The `agenticraft_foundation.mpst.patterns` module provides factories for these patterns, so you do not have to define them from scratch.

```python
from agenticraft_foundation.mpst.patterns import request_response, pipeline

# Request-response pattern
rr = request_response.create("client", "server", "Request", "Response")
print(f"Request-response roles: {rr.roles}")

# Pipeline pattern
pipe = pipeline.create(["ingester", "processor", "writer"], ["RawData", "ProcessedData"])
print(f"Pipeline roles: {pipe.roles}")
```

The `request_response` pattern creates a two-party protocol where one role sends a request and receives a response. The `pipeline` pattern chains multiple roles together, where each role receives from the previous one and sends to the next.

These patterns are fully verified `GlobalType` instances -- you can project them, check well-formedness, and create monitors from them just like any protocol you define manually.

## Complete Script

```python
"""MPST Verification Tutorial - Complete Script

Defines a RAG pipeline protocol as a multiparty session type,
verifies well-formedness, and sets up runtime monitoring.
"""
from agenticraft_foundation.mpst import (
    GlobalInteraction, GlobalType,
    Projector, SessionTypeChecker, SessionMonitor,
)
from agenticraft_foundation.mpst.patterns import request_response, pipeline

# Step 1: Define global type
rag_protocol = GlobalType(interactions=[
    GlobalInteraction(sender="client", receiver="retriever", message_type="Query"),
    GlobalInteraction(sender="retriever", receiver="generator", message_type="Context"),
    GlobalInteraction(sender="generator", receiver="client", message_type="Response"),
])

# Step 2: Project to local types
projector = Projector()
local_client = projector.project(rag_protocol, "client")
local_retriever = projector.project(rag_protocol, "retriever")
local_generator = projector.project(rag_protocol, "generator")

print(f"Client view: {local_client}")
print(f"Retriever view: {local_retriever}")
print(f"Generator view: {local_generator}")

# Step 3: Verify well-formedness
checker = SessionTypeChecker()
result = checker.check_well_formedness(rag_protocol)
print(f"Well-formed: {result.is_valid}")
if not result.is_valid:
    for error in result.errors:
        print(f"  Error: {error}")

# Step 4: Runtime monitoring
monitor = SessionMonitor(local_type=local_client)
monitor.send("Query", to="retriever")
monitor.receive("Response", from_role="generator")
print(f"Session complete: {monitor.is_complete()}")

# Step 5: Built-in patterns
rr = request_response.create("client", "server", "Request", "Response")
print(f"Request-response roles: {rr.roles}")

pipe = pipeline.create(["ingester", "processor", "writer"], ["RawData", "ProcessedData"])
print(f"Pipeline roles: {pipe.roles}")
```

## Next Steps

- Read [Session Types Concepts](../concepts/session-types.md) for the theoretical background on MPST
- Explore the [MPST API Reference](../api/mpst/index.md) for advanced features like choice types and recursion
- Continue to [Multi-Protocol Routing](protocol-routing.md) to learn how to route messages across protocol boundaries
