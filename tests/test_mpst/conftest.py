"""Test fixtures for MPST tests."""

from __future__ import annotations

import pytest

from agenticraft_foundation.mpst import (
    ChoiceType,
    MessageType,
    ParticipantId,
    Projector,
    RecursionType,
    SessionTypeChecker,
    WellFormednessChecker,
    choice,
    msg,
    rec,
    var,
)


@pytest.fixture
def client() -> ParticipantId:
    """Client participant."""
    return ParticipantId("client")


@pytest.fixture
def server() -> ParticipantId:
    """Server participant."""
    return ParticipantId("server")


@pytest.fixture
def coordinator() -> ParticipantId:
    """Coordinator participant."""
    return ParticipantId("coordinator")


@pytest.fixture
def worker1() -> ParticipantId:
    """Worker 1 participant."""
    return ParticipantId("worker1")


@pytest.fixture
def worker2() -> ParticipantId:
    """Worker 2 participant."""
    return ParticipantId("worker2")


@pytest.fixture
def request_response_type() -> MessageType:
    """Simple request-response global type.

    Client → Server : Request.
    Server → Client : Response.
    end
    """
    return msg(
        "client",
        "server",
        "request",
        msg("server", "client", "response"),
    )


@pytest.fixture
def choice_type() -> ChoiceType:
    """Global type with choice.

    Client → Server : {
        buy: Server → Client : Confirm. end,
        cancel: Server → Client : Cancelled. end
    }
    """
    return choice(
        "client",
        "server",
        {
            "buy": msg("server", "client", "confirm"),
            "cancel": msg("server", "client", "cancelled"),
        },
    )


@pytest.fixture
def recursive_type() -> RecursionType:
    """Recursive global type.

    μX. Client → Server : Request.
        Server → Client : Response.
        X
    """
    return rec(
        "X",
        msg(
            "client",
            "server",
            "request",
            msg("server", "client", "response", var("X")),
        ),
    )


@pytest.fixture
def projector() -> Projector:
    """Projector instance."""
    return Projector()


@pytest.fixture
def checker() -> SessionTypeChecker:
    """Session type checker instance."""
    return SessionTypeChecker()


@pytest.fixture
def well_formedness_checker() -> WellFormednessChecker:
    """Well-formedness checker instance."""
    return WellFormednessChecker()
