"""Tests for composable protocol transformers.

Covers:
- IdentityTransformer (T_{p→p} = id)
- ComposedTransformer (T_{p→p''} = T_{p'→p''} . T_{p→p'})
- TransformerRegistry (auto-composition, round-trip verification)
"""

from __future__ import annotations

import pytest

from agenticraft_foundation.protocols.transformers import (
    BaseProtocolTransformer,
    ComposedTransformer,
    IdentityTransformer,
    TransformerClassification,
    TransformerRegistry,
    TransformResult,
)
from agenticraft_foundation.types import ProtocolName


class SimpleTransformer(BaseProtocolTransformer):
    """Test transformer that passes messages through."""

    def __init__(self, src: ProtocolName, tgt: ProtocolName, prefix: str = ""):
        self._src = src
        self._tgt = tgt
        self._prefix = prefix

    @property
    def source_protocol(self) -> ProtocolName:
        return self._src

    @property
    def target_protocol(self) -> ProtocolName:
        return self._tgt

    @property
    def classification(self) -> TransformerClassification:
        return TransformerClassification.LOSSLESS

    def transform(self, message: dict) -> TransformResult:
        transformed = {f"{self._prefix}{k}" if self._prefix else k: v for k, v in message.items()}
        return TransformResult(
            success=True,
            message=transformed,
            source_protocol=self._src,
            target_protocol=self._tgt,
            fields_preserved=len(message),
        )


class TestIdentityTransformer:
    def test_identity_preserves_message(self):
        t = IdentityTransformer(ProtocolName.MCP)
        result = t.transform({"action": "test", "data": 42})
        assert result.success
        assert result.message == {"action": "test", "data": 42}

    def test_identity_is_lossless(self):
        t = IdentityTransformer(ProtocolName.MCP)
        assert t.classification == TransformerClassification.LOSSLESS

    def test_identity_same_protocol(self):
        t = IdentityTransformer(ProtocolName.A2A)
        assert t.source_protocol == t.target_protocol == ProtocolName.A2A


class TestComposedTransformer:
    def test_composition_works(self):
        t1 = SimpleTransformer(ProtocolName.MCP, ProtocolName.A2A)
        t2 = SimpleTransformer(ProtocolName.A2A, ProtocolName.ANP)
        composed = ComposedTransformer(t1, t2)
        assert composed.source_protocol == ProtocolName.MCP
        assert composed.target_protocol == ProtocolName.ANP

    def test_composition_type_mismatch_raises(self):
        t1 = SimpleTransformer(ProtocolName.MCP, ProtocolName.A2A)
        t2 = SimpleTransformer(ProtocolName.ANP, ProtocolName.CUSTOM)
        with pytest.raises(ValueError, match="Cannot compose"):
            ComposedTransformer(t1, t2)

    def test_composition_executes(self):
        t1 = SimpleTransformer(ProtocolName.MCP, ProtocolName.A2A)
        t2 = SimpleTransformer(ProtocolName.A2A, ProtocolName.ANP)
        composed = ComposedTransformer(t1, t2)
        result = composed.transform({"key": "value"})
        assert result.success
        assert result.message is not None

    def test_lossless_composition_stays_lossless(self):
        t1 = SimpleTransformer(ProtocolName.MCP, ProtocolName.A2A)
        t2 = SimpleTransformer(ProtocolName.A2A, ProtocolName.ANP)
        composed = ComposedTransformer(t1, t2)
        assert composed.classification == TransformerClassification.LOSSLESS


class TestTransformerRegistry:
    def test_identity_auto_registered(self):
        registry = TransformerRegistry()
        for p in ProtocolName:
            assert registry.has_transformer(p, p)

    def test_register_and_retrieve(self):
        registry = TransformerRegistry()
        t = SimpleTransformer(ProtocolName.MCP, ProtocolName.A2A)
        registry.register(t)
        retrieved = registry.get_direct_transformer(ProtocolName.MCP, ProtocolName.A2A)
        assert retrieved is t

    def test_auto_composition(self):
        registry = TransformerRegistry()
        registry.register(SimpleTransformer(ProtocolName.MCP, ProtocolName.A2A))
        registry.register(SimpleTransformer(ProtocolName.A2A, ProtocolName.ANP))
        composed = registry.get_transformer(ProtocolName.MCP, ProtocolName.ANP)
        assert composed is not None
        result = composed.transform({"test": 1})
        assert result.success

    def test_round_trip_identity(self):
        registry = TransformerRegistry()
        registry.register(SimpleTransformer(ProtocolName.MCP, ProtocolName.A2A))
        registry.register(SimpleTransformer(ProtocolName.A2A, ProtocolName.MCP))
        assert registry.verify_round_trip(
            ProtocolName.MCP,
            ProtocolName.A2A,
            {"action": "test"},
        )

    def test_get_all_paths(self):
        registry = TransformerRegistry()
        registry.register(SimpleTransformer(ProtocolName.MCP, ProtocolName.A2A))
        registry.register(SimpleTransformer(ProtocolName.A2A, ProtocolName.ANP))
        paths = registry.get_all_paths(ProtocolName.MCP, ProtocolName.ANP)
        assert len(paths) >= 1
        assert paths[0][0] == ProtocolName.MCP
        assert paths[0][-1] == ProtocolName.ANP

    def test_statistics(self):
        registry = TransformerRegistry()
        registry.register(SimpleTransformer(ProtocolName.MCP, ProtocolName.A2A))
        stats = registry.get_statistics()
        assert stats["direct_transformers"] == 1
        assert stats["identity_transformers"] == len(ProtocolName)
