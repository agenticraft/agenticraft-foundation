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
