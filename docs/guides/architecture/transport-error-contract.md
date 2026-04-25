# Transport error contract

What raises, what is logged, and what callers see when something goes
wrong on the wire. Pair with
[transport.md](transport.md) for the happy-path protocol and
[transport-tunables.md](transport-tunables.md) for the knobs that shape
the failure modes.

The contract is intentionally narrow: **two failure shapes
clients catch by type, plus the application's own exception** for
callback failures. Everything else is a bug and is logged with a
traceback.

---

## The exception hierarchy

Defined in
[`src/tinyros/transport/_errors.py`](../../../src/tinyros/transport/_errors.py)
and re-exported as `tinyros.TransportError` /
`tinyros.ConnectionLost` / `tinyros.SerializationError`:

```text
Exception
├── TransportError                                  # base; catch-all
│   ├── ConnectionLost (also ConnectionError)       # peer is gone
│   └── SerializationError                          # wire payload bad
└── (whatever the callback raised, on success-but-failed calls)
```

`ConnectionLost` multi-inherits from the builtin `ConnectionError`,
so callers written before this hierarchy that catch `ConnectionError`
keep working unchanged.

---

## Client-side: what futures resolve with

Every `client.call(...)` returns a `concurrent.futures.Future`. After
calling `.result()` on it, the caller sees one of:

| Outcome | Future state | What you got |
|---|---|---|
| Callback ran and returned a value | `result` set | The return value (pickled and back). |
| Callback raised | `exception` set | The original exception (pickled and back). |
| Callback raised something un-picklable | `exception` set | A synthesized `RuntimeError` naming the original type. |
| Encode failed before send (e.g., un-picklable arg) | `exception` set | `SerializationError`. |
| Connection dropped or never established | `exception` set | `ConnectionLost`. |
| Server is shutting down mid-call | `exception` set | `ConnectionLost`. |

**Recommended catch pattern:**

```python
fut = client.call("topic", payload)
try:
    result = fut.result(timeout=2.0)
except SerializationError:
    # The payload itself is the problem. Retrying will not help.
    ...
except ConnectionLost:
    # The peer is gone. Retry, reconnect, or escalate.
    ...
except concurrent.futures.TimeoutError:
    # The peer accepted the call but did not reply in time.
    ...
except Exception:
    # The remote callback raised. The exception you see here is
    # whatever the callback raised, after a pickle round-trip.
    ...
```

---

## Server-side: how failures are logged

Each row is a real code path; the log level reflects "is this the
operator's problem to fix?".

| Trigger | Log level | Action |
|---|---|---|
| Inbound CALL body is unparseable (`SerializationError`) | warning | Drop the connection. Peer sent garbage; not our bug. |
| Inbound CALL_LARGE metadata unparseable | error | Synthesize a failure REPLY for the caller, drop the conn. |
| Inbound CALL_LARGE shm materialization fails (segment missing) | error | Synthesize a failure REPLY, do not drop the conn. |
| Callback raises an exception | (unlogged on server, propagated to caller) | The exception is shipped back as part of the REPLY frame. |
| Callback's return value is un-picklable | warning | Substitute a `RuntimeError`; the caller sees that instead of hanging. |
| `ThreadPoolExecutor.submit` returns `RuntimeError` | (unlogged) | Pool is shutting down; drop the slot silently. |
| `ThreadPoolExecutor.submit` raises anything else | error + traceback | Re-raise after releasing the in-flight slot. This is a bug. |
| Reader handler raises `SerializationError` | warning | Drop the connection. |
| Reader handler raises anything else | error + traceback | Drop the connection. This is a bug. |
| Reply socket write fails (`OSError`) | (silent) | Connection is gone; nothing to do. |

The split between *warning* and *error* is the contract: warning lines
should not page anyone, error lines should be investigated.

---

## What does **not** raise to the caller

- **Reply-side `OSError`** when the socket is already closed -- the
  server discards quietly. The client's recv loop will fail every
  pending future with `ConnectionLost` once it notices the EOF.
- **Server-side traceback for callback exceptions** -- the server
  does not print a traceback when a callback raises; it ships the
  exception. If you want a server-side log too, log inside the
  callback before re-raising.

---

## Implementation pointers

- Raise sites for `ConnectionLost`:
  [`_client.py`](../../../src/tinyros/transport/_client.py).
- Raise site for `SerializationError`:
  [`_client.py`](../../../src/tinyros/transport/_client.py) (encode
  path) and
  [`_server.py`](../../../src/tinyros/transport/_server.py) (CALL
  decode path).
- Reader-loop classification:
  [`TinyServer._reader_loop`](../../../src/tinyros/transport/_server.py).
- Pool-submit classification:
  [`TinyServer._submit_call`](../../../src/tinyros/transport/_server.py).
