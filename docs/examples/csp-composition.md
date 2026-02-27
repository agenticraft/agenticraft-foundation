# CSP Composition

**Source:** `examples/csp_composition.py`

This example demonstrates all core CSP operators working together to model multi-agent coordination scenarios. It covers process construction, composition, hiding, recursion, and formal analysis.

---

## 1. Basic Processes

The example begins by defining events and a simple request-response handler.

```python
req = Event("request")
resp = Event("response")
ack = Event("ack")
internal = Event("internal_process")

# A simple request-response handler
handler = Prefix(req, Prefix(resp, Stop()))
print(f"Handler initials: {handler.initials()}")
print(f"Handler alphabet: {handler.alphabet()}")
```

`Prefix(req, Prefix(resp, Stop()))` builds a sequential process: perform `request`, then `response`, then deadlock (`Stop`). The `initials()` method returns the set of events the process can perform immediately. The `alphabet()` method returns all events the process can ever engage in.

## 2. Choice Operators

Two forms of choice model different decision-making scenarios.

```python
fast_path = Prefix(Event("cache_hit"), Prefix(resp, Stop()))
slow_path = Prefix(req, Prefix(internal, Prefix(resp, Stop())))

# External choice: environment picks
service = ExternalChoice(left=fast_path, right=slow_path)
print(f"Service (external choice) initials: {service.initials()}")

# Internal choice: process decides
strategy = InternalChoice(left=fast_path, right=slow_path)
print(f"Strategy (internal choice) initials: {strategy.initials()}")
```

**ExternalChoice** offers both branches to the environment (the caller decides which path to take). The initials include the first events of both branches. **InternalChoice** lets the process decide nondeterministically -- the environment cannot control which branch is taken. This distinction is fundamental in CSP: external choice models observable decisions, internal choice models hidden nondeterminism.

## 3. Parallel Composition

Two processes run concurrently, synchronizing on shared events.

```python
producer = Prefix(Event("produce"), Prefix(Event("emit"), Stop()))
consumer = Prefix(Event("emit"), Prefix(Event("consume"), Stop()))

# Synchronize on "emit"
system = Parallel(left=producer, right=consumer, sync_set=frozenset({Event("emit")}))
print(f"Parallel system alphabet: {system.alphabet()}")
```

The `sync_set` parameter specifies which events require both processes to participate simultaneously. Here, `emit` is a handoff -- the producer emits and the consumer receives at the same moment. Events not in the sync set (like `produce` and `consume`) can proceed independently.

## 4. Sequential Composition

One process runs after another completes.

```python
setup = Prefix(Event("init"), Skip())
work = Prefix(req, Prefix(resp, Stop()))
pipeline = Sequential(first=setup, second=work)
print(f"Sequential pipeline initials: {pipeline.initials()}")
```

The first process must terminate with `Skip()` (successful termination) before the second process begins. If the first process ends with `Stop()` (deadlock), the second process never starts. This is a key distinction: `Skip` signals completion, `Stop` signals failure.

## 5. Hiding

Internal events can be concealed from the external view.

```python
visible = Hiding(
    process=Prefix(internal, Prefix(resp, Stop())),
    hidden=frozenset({internal}),
)
print(f"After hiding 'internal': alphabet = {visible.alphabet()}")
```

Hiding converts visible events into internal (tau) transitions. The hidden event `internal` is removed from the alphabet. This is essential for abstraction -- you can build a detailed internal model and then hide implementation details when composing with other processes.

## 6. Recursion

Recursive definitions model infinite behavior like servers.

```python
server = Recursion(
    variable="X",
    body=Prefix(req, Prefix(resp, Variable("X"))),
)
unfolded = server.unfold()
print(f"Recursive server initials: {unfolded.initials()}")
```

`Recursion` defines a fixed-point: the process repeatedly performs `request` then `response`, forever. `Variable("X")` marks where the recursion unfolds. The `unfold()` method performs one step of unfolding, replacing the variable with the body.

## 7. Analysis

The example concludes with formal analysis of the constructed processes.

```python
lts = build_lts(handler)
print(f"Handler LTS: {len(lts.states)} states")

t = list(traces(lts, max_length=5))
print(f"Handler traces: {t}")

dl = detect_deadlock(lts)
print(f"Handler has deadlock: {dl.has_deadlock}")

print(f"Service is deadlock-free: {is_deadlock_free(service)}")
```

`build_lts()` constructs a Labeled Transition System -- a state machine representation of the process. `traces()` enumerates all possible execution sequences up to a given length. `detect_deadlock()` finds states with no outgoing transitions (where the process gets stuck). `is_deadlock_free()` is a convenience wrapper that returns a boolean.

The handler (`Prefix(req, Prefix(resp, Stop()))`) has a deadlock: after performing both events, it reaches `Stop`, which has no transitions. This is expected -- it models a one-shot handler, not a server.

---

??? example "Complete source"
    ```python
    """CSP Composition -- Core 8 operators working together.

    Demonstrates: Stop, Skip, Prefix, ExternalChoice, InternalChoice,
    Parallel, Sequential, Hiding, Recursion.
    """

    from agenticraft_foundation import (
        Event,
        ExternalChoice,
        Hiding,
        InternalChoice,
        Parallel,
        Prefix,
        Recursion,
        Sequential,
        Skip,
        Stop,
        Variable,
        build_lts,
        detect_deadlock,
        is_deadlock_free,
        traces,
    )

    # --- Events ---
    req = Event("request")
    resp = Event("response")
    ack = Event("ack")
    internal = Event("internal_process")

    # --- Basic processes ---
    # A simple request-response handler
    handler = Prefix(req, Prefix(resp, Stop()))
    print(f"Handler initials: {handler.initials()}")
    print(f"Handler alphabet: {handler.alphabet()}")

    # --- External Choice: environment picks ---
    fast_path = Prefix(Event("cache_hit"), Prefix(resp, Stop()))
    slow_path = Prefix(req, Prefix(internal, Prefix(resp, Stop())))
    service = ExternalChoice(left=fast_path, right=slow_path)
    print(f"\nService (external choice) initials: {service.initials()}")

    # --- Internal Choice: process decides ---
    strategy = InternalChoice(left=fast_path, right=slow_path)
    print(f"Strategy (internal choice) initials: {strategy.initials()}")

    # --- Parallel composition ---
    producer = Prefix(Event("produce"), Prefix(Event("emit"), Stop()))
    consumer = Prefix(Event("emit"), Prefix(Event("consume"), Stop()))
    # Synchronize on "emit"
    system = Parallel(left=producer, right=consumer, sync_set=frozenset({Event("emit")}))
    print(f"\nParallel system alphabet: {system.alphabet()}")

    # --- Sequential composition ---
    setup = Prefix(Event("init"), Skip())
    work = Prefix(req, Prefix(resp, Stop()))
    pipeline = Sequential(first=setup, second=work)
    print(f"Sequential pipeline initials: {pipeline.initials()}")

    # --- Hiding ---
    visible = Hiding(
        process=Prefix(internal, Prefix(resp, Stop())),
        hidden=frozenset({internal}),
    )
    print(f"After hiding 'internal': alphabet = {visible.alphabet()}")

    # --- Recursion ---
    server = Recursion(
        variable="X",
        body=Prefix(req, Prefix(resp, Variable("X"))),
    )
    unfolded = server.unfold()
    print(f"\nRecursive server initials: {unfolded.initials()}")

    # --- Analysis ---
    print("\n--- Analysis ---")
    lts = build_lts(handler)
    print(f"Handler LTS: {len(lts.states)} states")

    t = list(traces(lts, max_length=5))
    print(f"Handler traces: {t}")

    dl = detect_deadlock(lts)
    print(f"Handler has deadlock: {dl.has_deadlock}")

    print(f"Service is deadlock-free: {is_deadlock_free(service)}")
    ```
