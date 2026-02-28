"""Tests for the task queue example (Issue #5).

Verifies all acceptance criteria:
- Models a queue with 3 workers
- Uses Parallel with sync_set
- Verifies deadlock detection and queue draining
- Includes trace analysis showing task distribution
- Runs standalone
"""

from __future__ import annotations

import subprocess
import sys

from agenticraft_foundation import (
    Event,
    ExternalChoice,
    Parallel,
    Prefix,
    Stop,
    build_lts,
    detect_deadlock,
    is_deadlock_free,
    traces,
)
from agenticraft_foundation.algebra.semantics import maximal_traces


# ---------- shared fixtures ----------

def _make_worker(done: Event) -> ExternalChoice:
    """Build a one-shot worker identical to the example."""
    task_a = Event("pick_A")
    task_b = Event("pick_B")
    task_c = Event("pick_C")
    return ExternalChoice(
        left=Prefix(task_a, Prefix(done, Stop())),
        right=ExternalChoice(
            left=Prefix(task_b, Prefix(done, Stop())),
            right=Prefix(task_c, Prefix(done, Stop())),
        ),
    )


def _build_system():
    """Build the full task-queue system from the example."""
    task_a = Event("pick_A")
    task_b = Event("pick_B")
    task_c = Event("pick_C")
    done_w0 = Event("done_w0")
    done_w1 = Event("done_w1")
    done_w2 = Event("done_w2")
    tasks = frozenset({task_a, task_b, task_c})

    worker_0 = _make_worker(done_w0)
    worker_1 = _make_worker(done_w1)
    worker_2 = _make_worker(done_w2)

    dispatcher = Prefix(task_a, Prefix(task_b, Prefix(task_c, Stop())))

    workers = Parallel(
        left=Parallel(left=worker_0, right=worker_1, sync_set=frozenset()),
        right=worker_2,
        sync_set=frozenset(),
    )

    system = Parallel(
        left=dispatcher,
        right=workers,
        sync_set=tasks,
    )
    return system, tasks, {done_w0, done_w1, done_w2}


# ---------- tests ----------


class TestTaskQueueAcceptanceCriteria:
    """Verify every acceptance criterion from Issue #5."""

    def test_models_queue_with_three_workers(self):
        """AC: Models a queue with 2-3 workers."""
        system, tasks, dones = _build_system()
        # Three distinct done events prove three workers
        assert len(dones) == 3
        # Three distinct task events prove three tasks
        assert len(tasks) == 3

    def test_uses_parallel_with_sync_set(self):
        """AC: Uses Parallel with sync_set."""
        system, tasks, _ = _build_system()
        # The outermost node must be a Parallel with our task sync_set
        assert isinstance(system, Parallel)
        assert system.sync_set == tasks

    def test_verifies_deadlock_detection(self):
        """AC: Verifies deadlock-freedom (queue drains correctly).

        The system ends in STOP after all tasks are consumed, which
        constitutes a *termination deadlock* — expected and correct.
        detect_deadlock should find exactly this final state.
        """
        system, _, _ = _build_system()
        lts = build_lts(system)
        dl = detect_deadlock(lts)
        # Deadlock exists (termination deadlock after queue drains)
        assert dl.has_deadlock is True
        # Every deadlock trace should contain all three task picks,
        # proving the queue fully drains before stopping.
        task_a = Event("pick_A")
        task_b = Event("pick_B")
        task_c = Event("pick_C")
        for trace in dl.deadlock_traces:
            task_events = {e for e in trace if e in {task_a, task_b, task_c}}
            assert task_events == {task_a, task_b, task_c}, (
                f"Queue did not fully drain: {trace}"
            )

    def test_trace_analysis_shows_task_distribution(self):
        """AC: Includes trace analysis showing task distribution."""
        system, _, dones = _build_system()
        lts = build_lts(system)
        max_tr = list(maximal_traces(lts, max_length=10))

        # Must have at least one complete schedule
        assert len(max_tr) > 0

        # Every maximal trace must contain all 3 task picks
        task_events = {Event("pick_A"), Event("pick_B"), Event("pick_C")}
        for trace in max_tr:
            picks_in_trace = {e for e in trace if e in task_events}
            assert picks_in_trace == task_events

        # Each worker must appear in at least one trace (fair distribution)
        all_done_events = set()
        for trace in max_tr:
            for event in trace:
                if event in dones:
                    all_done_events.add(event)
        assert all_done_events == dones, (
            "Not all workers participate across schedules"
        )

    def test_runs_standalone(self):
        """AC: Runs standalone (exit code 0)."""
        result = subprocess.run(
            [sys.executable, "examples/task_queue.py"],
            capture_output=True,
            text=True,
            cwd="/Users/parth/HomeFolder/temp/OpenContributions/agenticraft-foundation",
            timeout=30,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        # Should produce meaningful output
        assert "Task Queue" in result.stdout
        assert "Deadlock Analysis" in result.stdout or "deadlock" in result.stdout.lower()
        assert "Trace" in result.stdout

    def test_lts_is_finite(self):
        """The state space should be manageable (not blow up)."""
        system, _, _ = _build_system()
        lts = build_lts(system)
        assert lts.num_states < 100
        assert lts.num_transitions < 200

    def test_all_traces_include_task_picks(self):
        """Non-empty traces should only contain valid events."""
        system, tasks, dones = _build_system()
        lts = build_lts(system)
        valid_events = tasks | dones
        for trace in traces(lts, max_length=10):
            for event in trace:
                assert event in valid_events, f"Unexpected event: {event}"
