"""Test fixtures for Process Algebra tests."""

from __future__ import annotations

import pytest

from agenticraft_foundation.algebra import (
    choice,
    parallel,
    prefix,
    rec,
    sequential,
    skip,
    var,
)


@pytest.fixture
def simple_prefix():
    """Simple prefix process: a → SKIP."""
    return prefix("a", skip())


@pytest.fixture
def choice_process():
    """External choice: (a → SKIP) □ (b → SKIP)."""
    return choice(prefix("a", skip()), prefix("b", skip()))


@pytest.fixture
def parallel_process():
    """Parallel process: (a → SKIP) ||| (b → SKIP)."""
    return parallel(prefix("a", skip()), prefix("b", skip()))


@pytest.fixture
def sync_parallel_process():
    """Synchronized parallel: (a → SKIP) |[{a}]| (a → SKIP)."""
    return parallel(prefix("a", skip()), prefix("a", skip()), {"a"})


@pytest.fixture
def sequential_process():
    """Sequential: (a → SKIP) ; (b → SKIP)."""
    return sequential(prefix("a", skip()), prefix("b", skip()))


@pytest.fixture
def recursive_process():
    """Recursive process: μX. a → X."""
    return rec("X", prefix("a", var("X")))


@pytest.fixture
def request_response_process():
    """Request-response: request → response → SKIP."""
    return prefix("request", prefix("response", skip()))


@pytest.fixture
def pipeline_process():
    """Pipeline: data_1 → data_2 → data_3 → SKIP."""
    return prefix("data_1", prefix("data_2", prefix("data_3", skip())))
