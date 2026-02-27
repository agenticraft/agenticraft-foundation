# Interrupt & Timeout

**Source:** `examples/interrupt_timeout.py`

This example demonstrates the 5 agent-specific CSP extensions through practical AI agent scenarios. Each operator addresses a real coordination problem that arises in multi-agent systems.

---

## 1. Interrupt: Priority Override

An agent performing a long-running task needs to be preemptible by higher-priority requests.

```python
process_data = Event("process_data")
return_result = Event("return_result")
handle_priority = Event("handle_priority")
urgent_response = Event("urgent_response")

# Normal task: process data, then return result
task = Prefix(process_data, Prefix(return_result, Stop()))

# Priority handler: handle urgent request
handler = Prefix(handle_priority, Prefix(urgent_response, Stop()))

# Interruptible task: can be preempted at any point
interruptible = Interrupt(primary=task, handler=handler)

print(f"Can do: {interruptible.initials()}")
# process_data (normal) OR handle_priority (interrupt)
```

The `Interrupt` operator wraps a primary process so that at any point during its execution, a handler process can take over. The initials include both the primary's next event (`process_data`) and the handler's trigger (`handle_priority`). After a normal event, the process remains interruptible. After the interrupt fires, the handler runs exclusively -- the primary is abandoned.

This models scenarios like: an agent analyzing a document can be interrupted by a high-priority user query.

## 2. Timeout: Bounded LLM Call

LLM API calls can hang indefinitely. A timeout ensures the system always makes progress.

```python
call_llm = Event("call_gpt4")
parse_response = Event("parse_response")
return_cached = Event("return_cached")

# LLM call process
llm_call = Prefix(call_llm, Prefix(parse_response, Stop()))

# Fallback: return cached response
fallback = Prefix(return_cached, Stop())

# Bounded execution: 30 second timeout
bounded = Timeout(process=llm_call, duration=30.0, fallback=fallback)

print(f"Initials: {bounded.initials()}")
# call_gpt4 (start LLM call) OR tau_timeout (timeout fires)
```

The `Timeout` operator adds a special `TIMEOUT_EVENT` (tau) to the initials. If the timeout fires (modeled as a tau transition), the fallback process takes over. If the primary process makes progress, the timeout remains active for subsequent steps. The duration is metadata for analysis -- the CSP semantics use the nondeterministic choice between progress and timeout.

After timeout, the fallback runs (`return_cached`). After the LLM call starts, the process remains a `Timeout` instance -- the timeout can still fire during `parse_response`.

## 3. Guard: Budget-Gated Agent

Expensive LLM calls should only proceed when budget allows.

```python
budget = {"remaining": 100.0}

expensive_call = Event("call_gpt4o")
process = Event("process_result")

expensive_agent = Prefix(expensive_call, Prefix(process, Stop()))

# Only activate if budget allows
guarded = Guard(
    condition=lambda: budget["remaining"] > 0,
    process=expensive_agent,
)

print(f"Budget > 0: initials = {guarded.initials()}")
# {call_gpt4o}

budget["remaining"] = 0
print(f"Budget = 0: initials = {guarded.initials()}")
# frozenset() -- acts as Stop

budget["remaining"] = 50.0
print(f"Budget restored: initials = {guarded.initials()}")
# {call_gpt4o} again
```

The `Guard` operator evaluates a condition (a boolean or callable). When the condition is true, the guarded process behaves normally. When false, it behaves like `Stop` -- no events are available. The condition can be a lambda that checks runtime state, making it dynamic.

This models resource-constrained agents: only activate expensive capabilities when budget, rate limits, or other conditions are met.

## 4. Rename: Protocol Bridging

Two agents use different event vocabularies. Rename translates between them.

```python
task_done = Event("task_done")
agent_a = Prefix(task_done, Stop())

work_complete = Event("work_complete")

# Bridge: rename task_done -> work_complete
bridged = Rename.from_dict(agent_a, {task_done: work_complete})

print(f"Original initials: {agent_a.initials()}")   # {task_done}
print(f"Bridged initials: {bridged.initials()}")     # {work_complete}
```

`Rename` applies a mapping from old events to new events across the entire process. The original event disappears from the alphabet and is replaced by the new one. `Rename.from_dict()` is a convenience constructor that accepts a dictionary mapping.

This models protocol bridging in heterogeneous multi-agent systems: Agent A emits `task_done` but Agent B expects `work_complete`. The rename adapter makes them compatible without modifying either agent.

## 5. Pipe: RAG Pipeline

A retrieval-augmented generation pipeline connects a retriever to a processor, hiding the internal channel.

```python
query = Event("query")
emit_docs = Event("emit_docs")
summarize = Event("summarize")

# Retriever: receives query, emits documents
retriever = Prefix(query, Prefix(emit_docs, Stop()))

# Processor: receives documents, produces summary
processor = Prefix(emit_docs, Prefix(summarize, Stop()))

# Pipeline: emit_docs is internal channel
rag_pipeline = Pipe(
    producer=retriever,
    consumer=processor,
    channel=frozenset({emit_docs}),
)

print(f"Pipeline alphabet: {rag_pipeline.alphabet()}")
# {query, summarize} -- emit_docs is hidden
```

`Pipe` composes two processes by synchronizing on channel events and then hiding those events from the external alphabet. The result is a pipeline where only the endpoints are visible: `query` goes in, `summarize` comes out, and `emit_docs` is an internal handoff.

This is equivalent to `Hiding(Parallel(retriever, processor, sync_set={emit_docs}), hidden={emit_docs})` but expressed as a single, intention-revealing operator.

## 6. Analysis

The example concludes by analyzing the constructed processes.

```python
lts = build_lts(interruptible)
t = list(traces(lts, max_length=5))
print(f"Interruptible traces ({len(t)}): {t[:5]}")

print(f"Timeout process deadlock-free: {is_deadlock_free(bounded)}")
print(f"Pipeline deadlock-free: {is_deadlock_free(rag_pipeline)}")
```

The interruptible process has more traces than the plain task because at each step, the interrupt can fire. The timeout process is not deadlock-free (it eventually reaches `Stop`), but it guarantees progress -- every path leads to a terminal action rather than getting stuck waiting for an LLM response. The pipeline is analyzed for deadlocks to verify the synchronization is correct.

---

??? example "Complete source"
    ```python
    """Interrupt, Timeout, Guard -- Agent-specific CSP extensions.

    Demonstrates the 5 new operators for real agent coordination scenarios.
    """

    from agenticraft_foundation import (
        TIMEOUT_EVENT,
        Event,
        Guard,
        Interrupt,
        Pipe,
        Prefix,
        Rename,
        Stop,
        Timeout,
        build_lts,
        is_deadlock_free,
        traces,
    )

    # =============================================================
    # Scenario 1: Interruptible Agent Task
    # =============================================================
    print("=== Interrupt: Priority Override ===")

    process_data = Event("process_data")
    return_result = Event("return_result")
    handle_priority = Event("handle_priority")
    urgent_response = Event("urgent_response")

    # Normal task: process data, then return result
    task = Prefix(process_data, Prefix(return_result, Stop()))

    # Priority handler: handle urgent request
    handler = Prefix(handle_priority, Prefix(urgent_response, Stop()))

    # Interruptible task: can be preempted at any point
    interruptible = Interrupt(primary=task, handler=handler)

    print(f"Can do: {interruptible.initials()}")
    # process_data (normal) OR handle_priority (interrupt)

    # After normal event, still interruptible
    after_normal = interruptible.after(process_data)
    print(f"After process_data: {after_normal}")
    print(f"Still interruptible: {isinstance(after_normal, Interrupt)}")

    # After interrupt event, handler takes over
    after_interrupt = interruptible.after(handle_priority)
    print(f"After interrupt: {after_interrupt}")
    print(f"Handler active: {not isinstance(after_interrupt, Interrupt)}")

    # =============================================================
    # Scenario 2: LLM Call with Timeout
    # =============================================================
    print("\n=== Timeout: Bounded LLM Call ===")

    call_llm = Event("call_gpt4")
    parse_response = Event("parse_response")
    return_cached = Event("return_cached")

    # LLM call process
    llm_call = Prefix(call_llm, Prefix(parse_response, Stop()))

    # Fallback: return cached response
    fallback = Prefix(return_cached, Stop())

    # Bounded execution: 30 second timeout
    bounded = Timeout(process=llm_call, duration=30.0, fallback=fallback)

    print(f"Initials: {bounded.initials()}")
    # call_gpt4 (start LLM call) OR tau_timeout (timeout fires)

    # If timeout fires, switch to fallback
    after_timeout = bounded.after(TIMEOUT_EVENT)
    print(f"After timeout: {after_timeout}")
    assert return_cached in after_timeout.initials()

    # If LLM starts, still has timeout
    after_start = bounded.after(call_llm)
    print(f"After call_llm: {after_start}")
    assert isinstance(after_start, Timeout)

    # =============================================================
    # Scenario 3: Conditional Agent Activation
    # =============================================================
    print("\n=== Guard: Budget-Gated Agent ===")

    budget = {"remaining": 100.0}

    expensive_call = Event("call_gpt4o")
    process = Event("process_result")

    expensive_agent = Prefix(expensive_call, Prefix(process, Stop()))

    # Only activate if budget allows
    guarded = Guard(
        condition=lambda: budget["remaining"] > 0,
        process=expensive_agent,
    )

    print(f"Budget > 0: initials = {guarded.initials()}")
    assert expensive_call in guarded.initials()

    # Deplete budget
    budget["remaining"] = 0
    print(f"Budget = 0: initials = {guarded.initials()}")
    assert guarded.initials() == frozenset()  # Acts as Stop

    # Restore budget
    budget["remaining"] = 50.0
    print(f"Budget restored: initials = {guarded.initials()}")
    assert expensive_call in guarded.initials()

    # =============================================================
    # Scenario 4: Protocol Bridging via Rename
    # =============================================================
    print("\n=== Rename: Protocol Bridging ===")

    # Agent A uses "task_done" event
    task_done = Event("task_done")
    agent_a = Prefix(task_done, Stop())

    # Agent B expects "work_complete" event
    work_complete = Event("work_complete")

    # Bridge: rename task_done -> work_complete
    bridged = Rename.from_dict(agent_a, {task_done: work_complete})

    print(f"Original initials: {agent_a.initials()}")
    print(f"Bridged initials: {bridged.initials()}")
    assert work_complete in bridged.initials()
    assert task_done not in bridged.initials()

    # =============================================================
    # Scenario 5: RAG Pipeline
    # =============================================================
    print("\n=== Pipe: RAG Pipeline ===")

    query = Event("query")
    emit_docs = Event("emit_docs")
    summarize = Event("summarize")

    # Retriever: receives query, emits documents
    retriever = Prefix(query, Prefix(emit_docs, Stop()))

    # Processor: receives documents, produces summary
    processor = Prefix(emit_docs, Prefix(summarize, Stop()))

    # Pipeline: emit_docs is internal channel
    rag_pipeline = Pipe(
        producer=retriever,
        consumer=processor,
        channel=frozenset({emit_docs}),
    )

    print(f"Pipeline alphabet: {rag_pipeline.alphabet()}")
    assert query in rag_pipeline.alphabet()
    assert summarize in rag_pipeline.alphabet()
    assert emit_docs not in rag_pipeline.alphabet()  # Hidden!

    # =============================================================
    # Analysis
    # =============================================================
    print("\n=== Analysis ===")

    lts = build_lts(interruptible)
    t = list(traces(lts, max_length=5))
    print(f"Interruptible traces ({len(t)}): {t[:5]}")

    print(f"Timeout process deadlock-free: {is_deadlock_free(bounded)}")
    print(f"Pipeline deadlock-free: {is_deadlock_free(rag_pipeline)}")

    print("\nAll examples passed.")
    ```
