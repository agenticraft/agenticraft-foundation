"""Request-Response pattern for client-server communication.

The simplest multi-party session pattern where a client sends a
request to a server and receives a response.

Global Type:
    Client → Server : Request.
    Server → Client : Response.
    end

This pattern is foundational for:
- API calls
- RPC mechanisms
- Query-answer protocols
"""

from __future__ import annotations

from dataclasses import dataclass

from agenticraft_foundation.mpst.global_types import (
    msg,
    rec,
    var,
)
from agenticraft_foundation.mpst.types import (
    ParticipantId,
    SessionType,
)


@dataclass
class RequestResponsePattern:
    """Request-Response session pattern.

    Attributes:
        client: Client participant ID
        server: Server participant ID
        request_label: Label for request message
        response_label: Label for response message
        repeatable: If True, pattern can repeat
    """

    client: ParticipantId | str = "client"
    server: ParticipantId | str = "server"
    request_label: str = "request"
    response_label: str = "response"
    repeatable: bool = False

    def __post_init__(self) -> None:
        self.client = ParticipantId(self.client)
        self.server = ParticipantId(self.server)

    def global_type(self) -> SessionType:
        """Build the global session type.

        Returns:
            Global type for request-response
        """
        # Client → Server : Request. Server → Client : Response. end
        response = msg(self.server, self.client, self.response_label)

        if self.repeatable:
            # μX. Client → Server : Request. Server → Client : Response. X
            body = msg(
                self.client,
                self.server,
                self.request_label,
                msg(self.server, self.client, self.response_label, var("X")),
            )
            return rec("X", body)

        return msg(self.client, self.server, self.request_label, response)

    def participants(self) -> set[ParticipantId]:
        """Get all participants in the request-response interaction.

        Returns:
            Set containing the client and server identifiers.
        """
        return {ParticipantId(self.client), ParticipantId(self.server)}


def request_response(
    client: str = "client",
    server: str = "server",
    request_label: str = "request",
    response_label: str = "response",
    repeatable: bool = False,
) -> SessionType:
    """Create a request-response global type.

    Args:
        client: Client participant name
        server: Server participant name
        request_label: Label for request
        response_label: Label for response
        repeatable: If True, pattern can repeat

    Returns:
        Global session type

    Example:
        # Simple request-response
        g = request_response("alice", "bob")

        # Repeatable RPC
        g = request_response("client", "api", repeatable=True)
    """
    pattern = RequestResponsePattern(
        client=client,
        server=server,
        request_label=request_label,
        response_label=response_label,
        repeatable=repeatable,
    )
    return pattern.global_type()


__all__ = [
    "RequestResponsePattern",
    "request_response",
]
