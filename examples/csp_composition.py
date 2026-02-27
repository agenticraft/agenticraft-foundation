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
