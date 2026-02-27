# Protocol Verification

**Source:** `examples/protocol_verification.py`

This example demonstrates Multiparty Session Types (MPST) for specifying and verifying communication protocols between agents. It covers global type definitions, projection to local types, and well-formedness checking.

---

## 1. Define a Request-Response Protocol

The simplest protocol: a client sends a request and receives a response.

```python
from agenticraft_foundation.mpst import (
    Projector,
    SessionTypeChecker,
    end,
    msg,
    project_all,
    request_response,
)

# Global type: Client sends request to Server, Server sends response to Client
protocol = request_response("client", "server")
print(f"Protocol: {protocol}")
```

The `request_response()` convenience function creates a global type with two message exchanges. A **global type** describes the entire protocol from a bird's-eye view -- it specifies who sends what to whom, and in what order. This is the specification that all participants must conform to.

## 2. Project to Local Types

Each participant gets a local view of the protocol.

```python
projector = Projector()
client_type = projector.project(protocol, "client")
server_type = projector.project(protocol, "server")
print(f"Client view: {client_type}")
print(f"Server view: {server_type}")
```

**Projection** extracts the local type for a single participant from the global type. The client's local type describes what the client does: send a request, then receive a response. The server's local type is the dual: receive a request, then send a response. Projection guarantees that if each participant follows their local type, the overall protocol is satisfied.

## 3. Check Well-Formedness

The protocol is validated for structural correctness.

```python
checker = SessionTypeChecker()
result = checker.check_well_formed(protocol)
print(f"Well-formed: {result.is_well_formed}")
```

**Well-formedness** ensures the global type is a valid protocol specification. The checker verifies properties such as:

- Every message has a defined sender and receiver.
- No participant sends a message to itself.
- The protocol terminates (reaches `end`).
- Branching is deterministic from the perspective of each participant.

A well-formed protocol can be safely projected and implemented.

## 4. Build a Custom 3-Party Protocol

A more complex RAG (retrieval-augmented generation) protocol with three participants.

```python
rag_protocol = msg(
    "client", "retriever", "query",
    msg(
        "retriever", "processor", "documents",
        msg(
            "processor", "client", "summary",
            end(),
        ),
    ),
)

print(f"RAG Protocol: {rag_protocol}")
```

The `msg()` function constructs a message exchange: `msg(sender, receiver, label, continuation)`. The RAG protocol specifies three steps:

1. The client sends a `query` to the retriever.
2. The retriever sends `documents` to the processor.
3. The processor sends a `summary` back to the client.

Each `msg()` call nests the continuation, building a sequential protocol. The `end()` at the bottom terminates the protocol.

## 5. Project and Validate the Custom Protocol

All participants get their local views, and the protocol is verified.

```python
all_projections = project_all(
    rag_protocol,
    participants=["client", "retriever", "processor"],
)

for participant, local_type in all_projections.items():
    print(f"  {participant}: {local_type}")

result = checker.check_well_formed(rag_protocol)
print(f"Well-formed: {result.is_well_formed}")
if not result.is_well_formed:
    for error in result.errors:
        print(f"  Error: {error}")
```

The `project_all()` convenience function projects the global type for every participant at once. Each local type describes that participant's obligations:

- **client**: Send `query` to retriever, then receive `summary` from processor.
- **retriever**: Receive `query` from client, then send `documents` to processor.
- **processor**: Receive `documents` from retriever, then send `summary` to client.

The well-formedness check validates the complete protocol. If errors are found, they are reported with details about which property was violated.

This approach enables protocol-first development: define the global type, verify it, project to local types, and implement each participant against its local type. The type system guarantees that the composed system conforms to the specification.

---

??? example "Complete source"
    ```python
    """Protocol Verification -- Multiparty Session Types.

    Demonstrates SessionMonitor and SessionTypeChecker for protocol conformance.
    """

    from agenticraft_foundation.mpst import (
        Projector,
        SessionTypeChecker,
        end,
        msg,
        project_all,
        request_response,
    )

    # =============================================================
    # Define a request-response protocol
    # =============================================================
    print("=== Request-Response Protocol ===")

    # Global type: Client sends request to Server, Server sends response to Client
    protocol = request_response("client", "server")
    print(f"Protocol: {protocol}")

    # Project to local types
    projector = Projector()
    client_type = projector.project(protocol, "client")
    server_type = projector.project(protocol, "server")
    print(f"Client view: {client_type}")
    print(f"Server view: {server_type}")

    # Verify well-formedness
    checker = SessionTypeChecker()
    result = checker.check_well_formed(protocol)
    print(f"Well-formed: {result.is_well_formed}")

    # =============================================================
    # Build a more complex protocol manually
    # =============================================================
    print("\n=== Custom Multi-Party Protocol ===")

    # Three-party protocol:
    # 1. Client sends "query" to Retriever
    # 2. Retriever sends "documents" to Processor
    # 3. Processor sends "summary" to Client
    rag_protocol = msg(
        "client", "retriever", "query",
        msg(
            "retriever", "processor", "documents",
            msg(
                "processor", "client", "summary",
                end(),
            ),
        ),
    )

    print(f"RAG Protocol: {rag_protocol}")

    # Project all participants
    all_projections = project_all(
        rag_protocol,
        participants=["client", "retriever", "processor"],
    )

    for participant, local_type in all_projections.items():
        print(f"  {participant}: {local_type}")

    # Verify
    result = checker.check_well_formed(rag_protocol)
    print(f"Well-formed: {result.is_well_formed}")
    if not result.is_well_formed:
        for error in result.errors:
            print(f"  Error: {error}")

    print("\nProtocol verification complete.")
    ```
