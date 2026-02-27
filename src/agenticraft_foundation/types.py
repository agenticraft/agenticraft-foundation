"""
Foundation Types for the AgentiCraft Mesh.

Core type definitions that sit at the foundation layer, enabling both
formal verification and runtime services to share common abstractions
without circular dependencies.

This module contains protocol-agnostic enumerations and types that are
used across the mesh foundation and services layers.

Version: 0.1.0
Date: 2025-12-20
"""

from __future__ import annotations

from enum import Enum


class ProtocolName(str, Enum):
    """
    Supported agent communication protocols.

    This foundational type enables protocol-aware routing, formal verification,
    and cross-protocol bridging throughout the mesh.

    Protocols:
        MCP: Model Context Protocol (Anthropic) - Agent-to-Tool communication
        A2A: Agent-to-Agent Protocol (Google) - Agent-to-Agent communication
        ANP: Agent Network Protocol (W3C) - Decentralized P2P networks
        CUSTOM: Custom protocol adapters for extension
    """

    MCP = "mcp"
    """Model Context Protocol - Agent-to-Tool interactions."""

    A2A = "a2a"
    """Agent-to-Agent Protocol - Agent coordination and task delegation."""

    ANP = "anp"
    """Agent Network Protocol - Decentralized peer-to-peer mesh."""

    CUSTOM = "custom"
    """Custom protocol adapter for extensibility."""


__all__ = [
    "ProtocolName",
]
