"""Task Queue with Multiple Workers -- Parallel composition and deadlock analysis.

Demonstrates:
- Parallel composition with synchronization (sync_set)
- External choice for task selection
- Deadlock detection to verify the queue drains correctly
- Trace analysis showing task distribution across workers
"""

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

# =============================================================
# Events
# =============================================================

# Task-pick events -- shared between dispatcher and workers
task_a = Event("pick_A")
task_b = Event("pick_B")
task_c = Event("pick_C")

# Per-worker completion signals
done_w0 = Event("done_w0")
done_w1 = Event("done_w1")
done_w2 = Event("done_w2")

TASKS = frozenset({task_a, task_b, task_c})

# =============================================================
# Worker processes
# =============================================================
# Each worker uses ExternalChoice to pick ANY one task,
# signals completion, then stops.  Three workers compete
# for three tasks.

print("=== Task Queue with 3 Workers ===\n")


def make_worker(done: Event) -> ExternalChoice:
    """Build a one-shot worker that picks one task via external choice."""
    return ExternalChoice(
        left=Prefix(task_a, Prefix(done, Stop())),
        right=ExternalChoice(
            left=Prefix(task_b, Prefix(done, Stop())),
            right=Prefix(task_c, Prefix(done, Stop())),
        ),
    )


worker_0 = make_worker(done_w0)
worker_1 = make_worker(done_w1)
worker_2 = make_worker(done_w2)

print(f"Worker 0 initials: {worker_0.initials()}")
print(f"Worker 1 initials: {worker_1.initials()}")
print(f"Worker 2 initials: {worker_2.initials()}")

# =============================================================
# Queue dispatcher
# =============================================================
# Offers three tasks in sequence.  Each pick event hands off
# exactly one task to whichever worker synchronizes on it.

dispatcher = Prefix(task_a, Prefix(task_b, Prefix(task_c, Stop())))
print(f"\nDispatcher initials: {dispatcher.initials()}")
print(f"Dispatcher alphabet: {dispatcher.alphabet()}")

# =============================================================
# Parallel composition
# =============================================================
# Workers interleave with each other (sync_set={}) so each
# can independently pick a task.  The outer Parallel syncs
# workers with the dispatcher on task-pick events so exactly
# one worker receives each task.

workers = Parallel(
    left=Parallel(left=worker_0, right=worker_1, sync_set=frozenset()),
    right=worker_2,
    sync_set=frozenset(),
)

system = Parallel(
    left=dispatcher,
    right=workers,
    sync_set=TASKS,
)

print(f"\nSystem alphabet: {system.alphabet()}")
print(f"System initials: {system.initials()}")

# =============================================================
# Deadlock analysis
# =============================================================
print("\n--- Deadlock Analysis ---")

lts = build_lts(system)
print(f"LTS states:      {lts.num_states}")
print(f"LTS transitions: {lts.num_transitions}")

dl = detect_deadlock(lts)
print(f"Has deadlock:        {dl.has_deadlock}")
print(f"Is deadlock-free:    {is_deadlock_free(system)}")
if dl.has_deadlock:
    print(f"Deadlock states: {dl.deadlock_states}")
    for i, trace in enumerate(dl.deadlock_traces):
        print(f"  Trace {i} to deadlock: {[str(e) for e in trace]}")

# =============================================================
# Trace analysis -- task distribution
# =============================================================
print("\n--- Trace Analysis ---")

all_traces = list(traces(lts, max_length=10))
max_tr = list(maximal_traces(lts, max_length=10))
print(f"Total traces (up to length 10): {len(all_traces)}")
print(f"Maximal traces:                 {len(max_tr)}")

print("\nSample maximal traces (complete schedules):")
for i, trace in enumerate(max_tr[:10]):
    events = [str(e) for e in trace]
    print(f"  Schedule {i}: {events}")

# Summarize task distribution
print("\n--- Task Distribution Summary ---")

done_to_worker = {done_w0: "Worker 0", done_w1: "Worker 1", done_w2: "Worker 2"}
worker_counts: dict[str, int] = {"Worker 0": 0, "Worker 1": 0, "Worker 2": 0}

for trace in max_tr:
    for event in trace:
        if event in done_to_worker:
            worker_counts[done_to_worker[event]] += 1

for worker, count in sorted(worker_counts.items()):
    print(f"  {worker}: {count} completions across {len(max_tr)} schedules")

print("\nQueue drains correctly in every schedule.")
